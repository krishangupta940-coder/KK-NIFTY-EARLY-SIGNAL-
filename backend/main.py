import os, math, threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel, Field
from kiteconnect import KiteConnect

app = FastAPI(title='KK NIFTY EARLY SIGNAL v3', version='4.0.0')
IST = ZoneInfo('Asia/Kolkata')
API_KEY = os.getenv('KITE_API_KEY', '')
API_SECRET = os.getenv('KITE_API_SECRET', '')
TOKEN_FILE = os.getenv('KITE_TOKEN_FILE', '/tmp/kite_access_token.txt')
_kite = None
_inputs = {}
_lock = threading.Lock()


def kite_client():
    global _kite
    if not API_KEY:
        return None
    if _kite is None:
        _kite = KiteConnect(api_key=API_KEY)
        token = os.getenv('KITE_ACCESS_TOKEN', '')
        if not token and os.path.exists(TOKEN_FILE):
            try:
                token = open(TOKEN_FILE).read().strip()
            except Exception:
                token = ''
        if token:
            _kite.set_access_token(token)
    return _kite


def set_token(token: str):
    global _kite
    if not API_KEY:
        raise HTTPException(503, 'KITE_API_KEY is not configured on Render')
    _kite = KiteConnect(api_key=API_KEY)
    _kite.set_access_token(token)
    try:
        with open(TOKEN_FILE, 'w') as f:
            f.write(token)
    except Exception:
        pass


@app.get('/')
def root():
    return {'app': 'KK NIFTY EARLY SIGNAL', 'version': '4.0.0', 'dashboard': '/app', 'docs': '/docs'}

@app.get('/health')
def health():
    return {'status': 'OK', 'version': '4.0.0', 'time_ist': datetime.now(IST).isoformat()}

@app.get('/zerodha/login')
def zerodha_login():
    if not API_KEY:
        raise HTTPException(503, 'KITE_API_KEY missing in Render environment')
    return RedirectResponse(kite_client().login_url())

@app.get('/zerodha/callback')
def zerodha_callback(request_token: str = '', status: str = '', action: str = ''):
    if status != 'success' or not request_token:
        raise HTTPException(400, 'Zerodha login cancelled or request_token missing')
    if not API_SECRET:
        raise HTTPException(503, 'KITE_API_SECRET missing in Render environment')
    try:
        data = kite_client().generate_session(request_token, api_secret=API_SECRET)
        set_token(data['access_token'])
        return RedirectResponse('/zerodha/success')
    except Exception as e:
        raise HTTPException(400, f'Zerodha session exchange failed: {str(e)}')

@app.get('/zerodha/success')
def zerodha_success():
    return {'status': 'connected', 'message': 'Zerodha session received. Open /app.'}

@app.get('/zerodha/status')
def zerodha_status():
    k = kite_client()
    if not k:
        return {'connected': False, 'reason': 'KITE_API_KEY not configured'}
    try:
        profile = k.profile()
        return {'connected': True, 'user_id': profile.get('user_id')}
    except Exception:
        return {'connected': False, 'reason': 'Login required. Open /zerodha/login.'}

@app.get('/market/nifty/status')
def nifty_status():
    k = kite_client()
    if not k:
        raise HTTPException(503, 'Zerodha API key not configured')
    try:
        q = k.ltp(['NSE:NIFTY 50'])['NSE:NIFTY 50']
        return {'symbol':'NIFTY 50', 'last_price':q['last_price'], 'instrument_token':q['instrument_token'], 'timestamp':datetime.now(IST).isoformat()}
    except Exception as e:
        raise HTTPException(503, f'Market data unavailable: {e}. Check /zerodha/login.')

@app.get('/market/candles')
def candles(interval: str = 'minute', days: int = 5):
    if interval not in ('minute','3minute','5minute'):
        raise HTTPException(400, 'interval must be minute, 3minute or 5minute')
    k = kite_client()
    if not k: raise HTTPException(503, 'Zerodha API not configured')
    try:
        end = datetime.now(IST)
        start = end - timedelta(days=max(1,min(days,30)))
        rows = k.historical_data(256265, start, end, interval)
        return {'candles': rows[-500:], 'count':len(rows)}
    except Exception as e:
        raise HTTPException(503, f'Historical data unavailable: {e}')


def indicators(df):
    c=df['close']; h=df['high']; l=df['low']; v=df['volume'].fillna(0)
    ema9=c.ewm(span=9, adjust=False).mean(); ema21=c.ewm(span=21, adjust=False).mean(); ema50=c.ewm(span=50, adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs=gain/(loss.replace(0,1e-9)); rsi=100-(100/(1+rs))
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean()
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    sig=macd.ewm(span=9,adjust=False).mean()
    return ema9,ema21,ema50,rsi,atr,macd,sig


def directional_prob(df, horizon):
    if len(df)<25:
        return {'call':None,'put':None,'no_trade':True,'reason':'कम से कम 25 candles उपलब्ध नहीं हैं'}
    e9,e21,e50,rsi,atr,macd,sig=indicators(df)
    i=-1; score=0.0
    score += 1.2 if e9.iloc[i]>e21.iloc[i] else -1.2
    score += 0.8 if e21.iloc[i]>e50.iloc[i] else -0.8
    score += 0.7 if macd.iloc[i]>sig.iloc[i] else -0.7
    score += 0.5 if rsi.iloc[i]>55 else (-0.5 if rsi.iloc[i]<45 else 0)
    lookback=3 if horizon==1 else 5
    slope=(e9.iloc[-1]-e9.iloc[-1-lookback])
    score += 0.6 if slope>0 else (-0.6 if slope<0 else 0)
    score=max(-3.8,min(3.8,score))
    # Transparent heuristic score, not a calibrated probability.
    pcall=round(100/(1+math.exp(-score*0.55)))
    pput=100-pcall
    confidence=abs(pcall-50)*2
    return {'call':pcall,'put':pput,'no_trade':confidence<18,'confidence_score':round(confidence), 'rsi':round(float(rsi.iloc[-1]),1), 'ema9':round(float(e9.iloc[-1]),2),'ema21':round(float(e21.iloc[-1]),2),'ema50':round(float(e50.iloc[-1]),2),'atr':round(float(atr.iloc[-1]),2) if pd.notna(atr.iloc[-1]) else None,'trend':'BULLISH' if pcall>=58 else ('BEARISH' if pput>=58 else 'SIDEWAYS'),'warning':'Trend weakening' if (e9.iloc[-1]>e21.iloc[-1] and slope<0) or (e9.iloc[-1]<e21.iloc[-1] and slope>0) else 'No reversal warning','method':'rule-based heuristic; not calibrated win probability'}

@app.get('/market/forecast')
def forecast():
    k=kite_client()
    if not k: raise HTTPException(503,'Zerodha not configured')
    try:
        end=datetime.now(IST); start=end-timedelta(days=5)
        rows=k.historical_data(256265,start,end,'minute')
        df=pd.DataFrame(rows)
        if df.empty: raise ValueError('No candle data returned')
        one=directional_prob(df,1); five=directional_prob(df,5)
        return {'timestamp':end.isoformat(),'one_minute':one,'five_minute':five,'disclaimer':'Probabilities are heuristic directional scores, not statistically validated odds. Do not trade on this alone.'}
    except Exception as e:
        raise HTTPException(503,f'Forecast unavailable: {e}')

class OvernightInputs(BaseModel):
    gift_nifty_pct: Optional[float]=Field(None, ge=-10, le=10)
    us_sp500_pct: Optional[float]=Field(None, ge=-10, le=10)
    us_nasdaq_pct: Optional[float]=Field(None, ge=-10, le=10)
    asian_markets_pct: Optional[float]=Field(None, ge=-10, le=10)
    crude_pct: Optional[float]=Field(None, ge=-30, le=30)
    usd_inr_pct: Optional[float]=Field(None, ge=-10, le=10)
    india_vix_pct: Optional[float]=Field(None, ge=-50, le=100)
    fii_cash_cr: Optional[float]=None
    fii_index_futures_cr: Optional[float]=None
    nifty_close_pct: Optional[float]=Field(None, ge=-10, le=10)

@app.post('/market/overnight-inputs')
def save_overnight_inputs(data: OvernightInputs):
    global _inputs
    _inputs=data.model_dump(exclude_none=True)
    return {'saved':True,'fields_received':list(_inputs.keys()),'note':'Values are user-supplied. Update them before market open/after close.'}

@app.get('/market/overnight')
def overnight():
    if not _inputs:
        return {'available':False,'required':'Enter available overnight inputs using POST /market/overnight-inputs','gap_up':None,'gap_down':None,'flat':None,'gap_size_pct':None}
    # Directional weighted proxy. Missing inputs are omitted and weights renormalized.
    weights={'gift_nifty_pct':0.35,'us_sp500_pct':0.12,'us_nasdaq_pct':0.10,'asian_markets_pct':0.12,'crude_pct':-0.06,'usd_inr_pct':-0.05,'india_vix_pct':-0.08,'fii_cash_cr':0.04,'fii_index_futures_cr':0.06,'nifty_close_pct':0.12}
    vals=[]
    for key,w in weights.items():
        if key in _inputs:
            val=float(_inputs[key])
            # FII values are scaled so crore figures do not dominate.
            if key.startswith('fii_'): val=max(-5000,min(5000,val))/1000
            vals.append((max(-5,min(5,val))*w,abs(w)))
    if not vals:
        return {'available':False,'gap_up':None,'gap_down':None,'flat':None,'gap_size_pct':None}
    score=sum(x for x,_ in vals)/sum(w for _,w in vals)
    score=max(-3,min(3,score))
    up=1/(1+math.exp(-score*1.25)); down=1/(1+math.exp(score*1.25)); flat=max(0.08,1-abs(up-down))
    total=up+down+flat; up,down,flat=[x/total for x in (up,down,flat)]
    return {'available':True,'gap_up':round(up*100,1),'gap_down':round(down*100,1),'flat':round(flat*100,1),'gap_size_pct':round(score*0.18,2),'directional_score':round(score,2),'inputs_used':list(_inputs.keys()),'note':'Indicative weighted proxy only; not calibrated. Confirm with live GIFT NIFTY and official data.'}

@app.get('/market/next-day')
def next_day():
    o=overnight()
    if not o.get('available'):
        return {'available':False,'message':'Overnight inputs are required. Next-day Bullish/Bearish/Sideways estimate is not available yet.'}
    return {'available':True,'bullish':o['gap_up'],'bearish':o['gap_down'],'sideways':o['flat'],'note':'Overnight proxy, not a calibrated next-day forecast.'}



class InstitutionalInputs(BaseModel):
    fii_cash_net_cr: Optional[float] = None
    dii_cash_net_cr: Optional[float] = None
    fii_index_futures_long: Optional[float] = None
    fii_index_futures_short: Optional[float] = None
    fii_index_options_call_oi: Optional[float] = None
    fii_index_options_put_oi: Optional[float] = None

_institutional = {}

@app.post('/market/institutional-inputs')
def save_institutional_inputs(data: InstitutionalInputs):
    global _institutional
    _institutional = data.model_dump(exclude_none=True)
    return {'saved': True, 'fields_received': list(_institutional.keys()),
            'note': 'FII/DII figures are manually supplied; update with the latest official exchange data.'}

@app.get('/market/institutional')
def institutional_flow():
    if not _institutional:
        return {'available': False, 'message': 'FII/DII figures not entered yet.'}
    cash = (_institutional.get('fii_cash_net_cr', 0) or 0) + (_institutional.get('dii_cash_net_cr', 0) or 0)
    longv = _institutional.get('fii_index_futures_long')
    shortv = _institutional.get('fii_index_futures_short')
    fut_bias = None
    if longv is not None and shortv is not None and (longv + shortv) > 0:
        fut_bias = round(100 * longv / (longv + shortv), 1)
    call_oi = _institutional.get('fii_index_options_call_oi')
    put_oi = _institutional.get('fii_index_options_put_oi')
    option_bias = None
    if call_oi is not None and put_oi is not None and (call_oi + put_oi) > 0:
        option_bias = round(100 * put_oi / (call_oi + put_oi), 1)
    return {'available': True, 'fii_dii_cash_net_cr': round(cash, 2),
            'fii_cash_net_cr': _institutional.get('fii_cash_net_cr'),
            'dii_cash_net_cr': _institutional.get('dii_cash_net_cr'),
            'fii_futures_long_share_pct': fut_bias,
            'fii_put_oi_share_pct': option_bias,
            'interpretation': 'Positive combined cash flow can support sentiment; futures/options positioning is context, not a direct buy/sell signal.',
            'source_note': 'Manual input; not a live institutional transaction feed.'}

@app.get('/market/big-move')
def big_move():
    k = kite_client()
    if not k:
        raise HTTPException(503, 'Zerodha not configured')
    try:
        end = datetime.now(IST)
        start = end - timedelta(days=5)
        rows = k.historical_data(256265, start, end, 'minute')
        df = pd.DataFrame(rows)
        if len(df) < 30:
            return {'available': False, 'message': 'Not enough intraday candles yet.'}
        e9, e21, e50, rsi, atr, macd, sig = indicators(df)
        recent = df.tail(20)
        ret20 = (float(recent.close.iloc[-1]) / float(recent.close.iloc[0]) - 1) * 100
        vol_now = float(recent.volume.tail(5).mean())
        vol_base = float(recent.volume.tail(20).mean())
        vol_ratio = vol_now / vol_base if vol_base > 0 else 1.0
        trend_score = 0
        trend_score += 1 if e9.iloc[-1] > e21.iloc[-1] else -1
        trend_score += 1 if e21.iloc[-1] > e50.iloc[-1] else -1
        trend_score += 1 if macd.iloc[-1] > sig.iloc[-1] else -1
        trend_score += 1 if ret20 > 0.08 else (-1 if ret20 < -0.08 else 0)
        trend_score += 1 if vol_ratio > 1.25 and ret20 > 0 else (-1 if vol_ratio > 1.25 and ret20 < 0 else 0)
        direction = 'BULLISH' if trend_score >= 2 else ('BEARISH' if trend_score <= -2 else 'UNCONFIRMED')
        strength = min(95, max(5, int(50 + trend_score * 9)))
        return {'available': True, 'window_minutes': 20, 'direction': direction,
                'strength_score': strength, 'return_last_20m_pct': round(ret20, 3),
                'volume_ratio': round(vol_ratio, 2), 'rsi': round(float(rsi.iloc[-1]), 1),
                'alert': 'HIGH ALERT' if abs(trend_score) >= 4 and vol_ratio >= 1.25 else ('EARLY WARNING' if abs(trend_score) >= 2 else 'NO CLEAR BIG MOVE'),
                'note': 'Heuristic momentum alert, not a prediction that a large move is certain. Historical candles are polled; not a true 1–2 second WebSocket feed.'}
    except Exception as e:
        raise HTTPException(503, f'Big-move analysis unavailable: {e}')


@app.get('/app', response_class=HTMLResponse)
def dashboard():
    return HTMLResponse('''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>KK NIFTY EARLY SIGNAL</title><style>body{font-family:Arial;background:#101827;color:#eef2ff;margin:0;padding:16px}.wrap{max-width:760px;margin:auto}h2{margin-bottom:4px}.muted{color:#aab6ca;font-size:13px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.card{background:#1d293b;padding:16px;border-radius:14px;margin:12px 0}.big{font-size:30px;font-weight:bold}.green{color:#4ade80}.red{color:#fb7185}button{background:#2563eb;color:white;border:0;border-radius:9px;padding:12px;width:100%;font-size:16px}input{width:100%;box-sizing:border-box;padding:10px;margin:5px 0;border-radius:8px;border:1px solid #475569;background:#0f172a;color:white}.row{display:flex;justify-content:space-between;gap:10px}</style></head><body><div class="wrap"><h2>KK NIFTY EARLY SIGNAL v3</h2><div class="muted">Market analysis dashboard · No auto trading · 5s refresh</div><div class="card"><div class="row"><b>NIFTY 50 LTP</b><span id="ltp">--</span></div><div class="muted" id="time">Waiting for market data</div></div><div class="grid"><div class="card"><b>1 Minute</b><div class="big" id="one">--</div><div id="oneput" class="muted">CALL / PUT</div></div><div class="card"><b>5 Minutes</b><div class="big" id="five">--</div><div id="fiveput" class="muted">CALL / PUT</div></div></div><div class="card"><b>15–20 Minute BIG MOVE Monitor</b><p id="bigmove">Waiting for live market candles.</p><p class="muted">5-second dashboard refresh; underlying candles and Kite data availability determine freshness.</p></div><div class="card"><b>FII / DII Flow</b><p id="fii">FII/DII figures not entered</p><p class="muted">Enter official figures; cash market data is generally published after market close.</p></div><div class="card"><b>Overnight Gap Forecast</b><p id="gap">Global inputs not entered yet.</p><p class="muted">Gap Up / Gap Down / Flat + estimated gap size</p></div><div class="card"><b>Next Day Outlook</b><p id="next">Awaiting overnight inputs.</p></div><div class="card"><b>Connection</b><p id="conn">Checking Zerodha...</p><button onclick="location.href='/zerodha/login'">Connect / Reconnect Zerodha</button></div><div class="card"><b>Overnight data entry</b><div class="muted">Enter percentage changes from previous close. Leave unknown fields blank.</div><div class="grid"><label>GIFT NIFTY %<input id="gift_nifty_pct" type="number" step="0.01"></label><label>US S&P 500 %<input id="us_sp500_pct" type="number" step="0.01"></label><label>Nasdaq %<input id="us_nasdaq_pct" type="number" step="0.01"></label><label>Asian markets %<input id="asian_markets_pct" type="number" step="0.01"></label><label>Crude change %<input id="crude_pct" type="number" step="0.01"></label><label>USD/INR change %<input id="usd_inr_pct" type="number" step="0.01"></label><label>India VIX change %<input id="india_vix_pct" type="number" step="0.01"></label><label>NIFTY close change %<input id="nifty_close_pct" type="number" step="0.01"></label></div><button onclick="saveInputs()">Save overnight inputs</button><p class="muted" id="save"></p></div><p class="muted">Percentages are rule-based estimates, not validated probabilities or guaranteed predictions. Verify all data and use risk limits.</p></div><script>async function get(u){let r=await fetch(u);return await r.json()}async function refresh(){try{let s=await get('/zerodha/status');document.getElementById('conn').textContent=s.connected?'Connected: '+(s.user_id||'Zerodha'):'Not connected — tap Connect';if(s.connected){let q=await get('/market/nifty/status');document.getElementById('ltp').textContent=q.last_price;document.getElementById('time').textContent=q.timestamp;let f=await get('/market/forecast');for(let [id,x] of [['one',f.one_minute],['five',f.five_minute]]){document.getElementById(id).textContent=x.call==null?'N/A':(x.trend+' '+x.call+'%');document.getElementById(id+'put').textContent=x.call==null?x.reason:('PUT '+x.put+'% · '+(x.no_trade?'NO TRADE':'Signal available')+' · '+x.warning)} }let bm=await get('/market/big-move');document.getElementById('bigmove').textContent=bm.available?(bm.alert+' · '+bm.direction+' · Strength '+bm.strength_score+'/100 · 20m change '+bm.return_last_20m_pct+'% · Vol '+bm.volume_ratio+'x'):bm.message;let fi=await get('/market/institutional');document.getElementById('fii').textContent=fi.available?('FII cash '+fi.fii_cash_net_cr+' cr · DII cash '+fi.dii_cash_net_cr+' cr · FII futures long share '+fi.fii_futures_long_share_pct+'%'):'FII/DII figures not entered';let o=await get('/market/overnight');document.getElementById('gap').textContent=o.available?('Gap Up '+o.gap_up+'% · Gap Down '+o.gap_down+'% · Flat '+o.flat+'% · Estimated gap '+o.gap_size_pct+'%'):'Inputs required';let n=await get('/market/next-day');document.getElementById('next').textContent=n.available?('Bullish '+n.bullish+'% · Bearish '+n.bearish+'% · Sideways '+n.sideways+'%'):'Not available until overnight inputs are entered.'}catch(e){document.getElementById('conn').textContent='Data unavailable: '+e.message}}async function saveInputs(){let keys=['gift_nifty_pct','us_sp500_pct','us_nasdaq_pct','asian_markets_pct','crude_pct','usd_inr_pct','india_vix_pct','nifty_close_pct'];let d={};for(let k of keys){let v=document.getElementById(k).value;if(v!=='')d[k]=Number(v)}let r=await fetch('/market/overnight-inputs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});document.getElementById('save').textContent=r.ok?'Saved. Forecast updating...':'Could not save inputs';refresh()}refresh();setInterval(refresh,5000);</script></body></html>''')
