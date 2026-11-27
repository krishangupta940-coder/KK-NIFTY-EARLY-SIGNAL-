import os, math, threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional
import pandas as pd
import requests
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel, Field
from kiteconnect import KiteConnect
def _date_sort_key(value):
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            pass
    return datetime.min
app = FastAPI(title='KK NIFTY EARLY SIGNAL v5', version='5.0.0')
IST = ZoneInfo('Asia/Kolkata')
API_KEY = os.getenv('KITE_API_KEY', '')
API_SECRET = os.getenv('KITE_API_SECRET', '')
TOKEN_FILE = os.getenv('KITE_TOKEN_FILE', '/tmp/kite_access_token.txt')  # Render local files may be lost on restart; set KITE_ACCESS_TOKEN again if needed.
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
    return {'app': 'KK NIFTY EARLY SIGNAL', 'version': '5.0.0', 'dashboard': '/app', 'docs': '/docs'}

@app.get('/health')
def health():
    return {'status': 'OK', 'version': '5.0.0', 'time_ist': datetime.now(IST).isoformat()}

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
    """Rule-based directional signal with trend, momentum and volume checks."""

    if df is None or len(df) < 55:
        return {
            'call': None,
            'put': None,
            'no_trade': True,
            'reason': 'कम से कम 55 candles जरूरी हैं',
            'horizon_minutes': horizon
        }

    try:
        e9, e21, e50, rsi, atr, macd, sig = indicators(df)

        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].fillna(0)

        lookback = max(2, min(int(horizon), 20))

        price = float(close.iloc[-1])
        prev_price = float(close.iloc[-lookback - 1])

        momentum = ((price / prev_price) - 1) * 100
        slope = float(e9.iloc[-1] - e9.iloc[-lookback])

        score = 0.0

        # EMA trend
        score += 1.2 if e9.iloc[-1] > e21.iloc[-1] else -1.2
        score += 0.8 if e21.iloc[-1] > e50.iloc[-1] else -0.8

        # MACD trend
        score += 0.7 if macd.iloc[-1] > sig.iloc[-1] else -0.7

        # RSI
        r = float(rsi.iloc[-1])
        if r > 55:
            score += 0.5
        elif r < 45:
            score -= 0.5

        # EMA movement
        score += 0.6 if slope > 0 else (-0.6 if slope < 0 else 0)

        # Momentum
        score += 0.6 if momentum > 0.03 else (
            -0.6 if momentum < -0.03 else 0
        )

        # Candle strength
        candle_range = max(float(high.iloc[-1] - low.iloc[-1]), 1e-9)
        body = float(close.iloc[-1] - close.iloc[-2])

        if body > 0 and close.iloc[-1] > close.iloc[-2]:
            score += 0.3
        elif body < 0 and close.iloc[-1] < close.iloc[-2]:
            score -= 0.3

        # Volume confirmation
        avg_volume = float(volume.tail(20).mean())
        current_volume = float(volume.iloc[-1])

        if avg_volume > 0 and current_volume > avg_volume * 1.2:
            score += 0.3 if close.iloc[-1] > close.iloc[-2] else -0.3

        # Limit score
        score = max(-4.7, min(4.7, score))

        # Directional score (not actual statistical probability)
        call_score = round(100 / (1 + math.exp(-score * 0.45)))
        put_score = 100 - call_score

        confidence = abs(call_score - 50) * 2

        trend = (
            'BULLISH' if call_score >= 58
            else 'BEARISH' if put_score >= 58
            else 'SIDEWAYS'
        )

        weakening = (
            (e9.iloc[-1] > e21.iloc[-1] and slope < 0)
            or (e9.iloc[-1] < e21.iloc[-1] and slope > 0)
        )

        no_trade = confidence < 30 or trend == 'SIDEWAYS' or weakening

        return {
            'horizon_minutes': horizon,
            'call': call_score,
            'put': put_score,
            'no_trade': no_trade,
            'confidence_score': round(confidence),
            'rsi': round(r, 1),
            'ema9': round(float(e9.iloc[-1]), 2),
            'ema21': round(float(e21.iloc[-1]), 2),
            'ema50': round(float(e50.iloc[-1]), 2),
            'atr': round(float(atr.iloc[-1]), 2)
                if pd.notna(atr.iloc[-1]) else None,
            'momentum_pct': round(momentum, 4),
            'trend': trend,
            'warning': 'Trend weakening' if weakening
                else 'No reversal warning',
            'method': 'Multi-indicator heuristic score; not a calibrated probability'
        }

    except Exception as e:
        return {
            'call': None,
            'put': None,
            'no_trade': True,
            'reason': f'Analysis error: {str(e)}',
            'horizon_minutes': horizon
        }


@app.get('/market/forecast')
def forecast():
    k = kite_client()
    if not k:
        raise HTTPException(503, 'Zerodha not configured')
    try:
        end = datetime.now(IST)
        start = end - timedelta(days=5)
        rows = k.historical_data(256265, start, end, 'minute')
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError('No candle data returned')
        return {
            'timestamp': end.isoformat(),
            'candle_count': len(df),
            'one_minute': directional_prob(df, 1),
            'three_minute': directional_prob(df, 3),
            'five_minute': directional_prob(df, 5),
            'fifteen_minute': directional_prob(df, 15),
            'disclaimer': 'CALL/PUT figures are heuristic directional scores, not statistically validated probabilities. Use paper trading and risk limits.'
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f'Forecast unavailable: {e}')


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


# Automatic overnight/global-market inputs.
# Yahoo Finance chart data is indicative and may be delayed or unavailable.
_YAHOO_CHART_URL = 'https://query1.finance.yahoo.com/v8/finance/chart/{}'

def _yahoo_daily_change(symbol):
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
    response = requests.get(
        _YAHOO_CHART_URL.format(requests.utils.quote(symbol, safe='')),
        params={'range': '5d', 'interval': '1d'},
        headers=headers, timeout=12
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get('chart', {}).get('result') or []
    if not result:
        raise ValueError('No Yahoo Finance data for ' + symbol)
    item = result[0]
    quote = item.get('indicators', {}).get('quote', [{}])[0]
    closes = [x for x in quote.get('close', []) if isinstance(x, (int, float)) and x > 0]
    if len(closes) < 2:
        raise ValueError('Not enough daily closes for ' + symbol)
    previous, latest = closes[-2], closes[-1]
    change = (latest / previous - 1) * 100
    return round(change, 3), item.get('meta', {}).get('currency')

def _automatic_global_inputs():
    symbols = {
        'us_sp500_pct': '^GSPC',
        'us_nasdaq_pct': '^IXIC',
        'crude_pct': 'BZ=F',
        'usd_inr_pct': 'INR=X',
        'india_vix_pct': '^INDIAVIX',
        'nifty_close_pct': '^NSEI',
    }
    values, errors = {}, {}
    for field, symbol in symbols.items():
        try:
            values[field], _ = _yahoo_daily_change(symbol)
        except Exception as exc:
            errors[field] = str(exc)[:180]

    asian_symbols = ['^N225', '^HSI', '000001.SS', '^KS11']
    asian_changes = []
    for symbol in asian_symbols:
        try:
            change, _ = _yahoo_daily_change(symbol)
            asian_changes.append(change)
        except Exception as exc:
            errors['asian_' + symbol] = str(exc)[:120]
    if asian_changes:
        values['asian_markets_pct'] = round(sum(asian_changes) / len(asian_changes), 3)

    return {
        'available': bool(values),
        'values': values,
        'errors': errors,
        'gift_nifty_pct': None,
        'gift_nifty_note': 'GIFT NIFTY is not filled: no stable free source configured.',
        'source': 'Yahoo Finance chart endpoint; indicative daily closes, may be delayed',
        'fetched_at': datetime.now(IST).isoformat(),
        'asian_indices_used': len(asian_changes),
        'note': 'Values are daily percentage changes from the previous available close, not guaranteed live quotes. Check timestamps and verify before use.'
    }

@app.get('/market/overnight-auto')
def overnight_auto():
    global _inputs
    result = _automatic_global_inputs()
    if result['available']:
        # Merge only successfully fetched fields; preserve any manually entered GIFT NIFTY.
        _inputs.update(result['values'])
    result['saved_fields'] = list(result['values'].keys())
    return result


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




# NSE daily FII/FPI and DII cash-market activity.
# NSE's public endpoint is not a guaranteed API; access can be rate-limited or changed.
_NSE_FII_DII_URL = 'https://www.nseindia.com/api/fiidiiTradeReact'
_FII_HISTORY_FILE = os.getenv(
    'FII_HISTORY_FILE',
    '/var/data/kk_fii_dii_history.json'
)
_fii_cache = {'updated_at': None, 'rows': None}
_fii_lock = threading.Lock()

def _num(value):
    try:
        return float(str(value).replace(',', '').replace('₹', '').strip())
    except Exception:
        return None

def _nse_fii_dii_fetch():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.nseindia.com/reports/fii-dii',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    session = requests.Session()
    session.headers.update(headers)
    # Establish cookies before calling NSE's report endpoint.
    session.get('https://www.nseindia.com/', timeout=12)
    response = session.get(_NSE_FII_DII_URL, timeout=15)
    response.raise_for_status()
    payload = response.json()
    records = payload if isinstance(payload, list) else payload.get('data', [])
    parsed = []
    for item in records:
        if not isinstance(item, dict):
            continue
        category = str(item.get('category', item.get('type', item.get('investorType', '')))).upper()
        if not any(tag in category for tag in ('FII', 'FPI', 'DII')):
            continue
        buy = _num(item.get('buyValue', item.get('buy', item.get('buy_value'))))
        sell = _num(item.get('sellValue', item.get('sell', item.get('sell_value'))))
        net = _num(item.get('netValue', item.get('net', item.get('net_value'))))
        if net is None and buy is not None and sell is not None:
            net = buy - sell
        if net is None:
            continue
        parsed.append({
            'date': item.get('date', item.get('tradingDate', datetime.now(IST).date().isoformat())),
            'category': 'FII' if ('FII' in category or 'FPI' in category) else 'DII',
            'buy_cr': buy, 'sell_cr': sell, 'net_cr': net
        })
    if not parsed:
        raise ValueError('NSE response format changed or contains no FII/DII rows')
    return parsed

def _load_fii_history():
    try:
        with open(_FII_HISTORY_FILE, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def _save_fii_history(rows):
    try:
        with open(_FII_HISTORY_FILE, 'w') as f:
            json.dump(rows[-30:], f)
    except Exception:
        pass

def _get_fii_dii_data(force=False):
    now = datetime.now(IST)
    with _fii_lock:
        updated = _fii_cache.get('updated_at')
        if not force and updated and (now - updated).total_seconds() < 900 and _fii_cache.get('rows') is not None:
            return _fii_cache['rows'], 'cache'
        latest = _nse_fii_dii_fetch()
        history = _load_fii_history()
        # One record per date/category; latest NSE value replaces same-day provisional value.
        by_key = {(str(x.get('date')), x.get('category')): x for x in history if isinstance(x, dict)}
        for row in latest:
            by_key[(str(row['date']), row['category'])] = row
        history = sorted(
    by_key.values(),
    key=lambda x: _date_sort_key(x.get('date')),
    reverse=True
)[:30]
        _save_fii_history(history)
        _fii_cache['updated_at'] = now
        _fii_cache['rows'] = history
        return history, 'NSE India official report endpoint'

@app.get('/market/institutional')
def institutional_flow():
    try:
        history, source = _get_fii_dii_data()
        dates = sorted(
    {str(x.get('date')) for x in history},
    key=_date_sort_key,
    reverse=True
        )
        last_dates = dates[:5]
        daily = []
        for day in last_dates:
            day_rows = [x for x in history if str(x.get('date')) == day]
            fii = next((x for x in day_rows if x.get('category') == 'FII'), None)
            dii = next((x for x in day_rows if x.get('category') == 'DII'), None)
            daily.append({
                'date': day,
                'fii_buy_cr': fii.get('buy_cr') if fii else None,
                'fii_sell_cr': fii.get('sell_cr') if fii else None,
                'fii_net_cr': fii.get('net_cr') if fii else None,
                'dii_buy_cr': dii.get('buy_cr') if dii else None,
                'dii_sell_cr': dii.get('sell_cr') if dii else None,
                'dii_net_cr': dii.get('net_cr') if dii else None,
                'combined_net_cr': round((fii.get('net_cr') or 0) + (dii.get('net_cr') or 0), 2) if fii or dii else None
            })
        total_fii = sum(x['fii_net_cr'] or 0 for x in daily if x['fii_net_cr'] is not None)
        total_dii = sum(x['dii_net_cr'] or 0 for x in daily if x['dii_net_cr'] is not None)
        combined = total_fii + total_dii
        if len(daily) < 2:
            bias = 'INSUFFICIENT HISTORY'
        elif combined > 0:
            bias = 'BULLISH BIAS'
        elif combined < 0:
            bias = 'BEARISH BIAS'
        else:
            bias = 'MIXED / NEUTRAL'
        return {
            'available': bool(daily), 'source': source,
            'latest_date': last_dates[0] if last_dates else None,
            'history_days_available': len(daily), 'target_history_days': 5,
            'daily': daily, 'five_day_fii_net_cr': round(total_fii, 2),
            'five_day_dii_net_cr': round(total_dii, 2),
            'five_day_combined_net_cr': round(combined, 2),
            'sentiment_bias': bias,
            'note': 'NSE cash-market FII/FPI and DII activity is provisional and generally published after market close. Five-day history is accumulated by this service; after a fresh deployment/restart it may take up to five trading sessions to rebuild. Not an intraday flow feed or a CALL/PUT signal.'
        }
    except Exception as e:
        return {
            'available': False, 'source': 'NSE India',
            'message': 'NSE FII/DII data could not be fetched right now.',
            'detail': str(e),
            'fallback': 'Try again later; do not enter zero values as a substitute.'
        }

@app.post('/market/institutional-refresh')
def institutional_refresh():
    _fii_cache['updated_at'] = None
    _fii_cache['rows'] = None
    return institutional_flow()

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
        recent15 = df.tail(15)
        ret15 = (float(recent15.close.iloc[-1]) / float(recent15.close.iloc[0]) - 1) * 100
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
        bullish_pct = min(95, max(5, int(50 + trend_score * 12.5)))
        bearish_pct = 100 - bullish_pct
        strength = max(bullish_pct, bearish_pct)
        alert = ('HIGH ALERT' if abs(trend_score) >= 4 and vol_ratio >= 1.25
                 else ('EARLY WARNING' if abs(trend_score) >= 2 else 'NO CLEAR BIG MOVE'))
        return {'available': True, 'window_minutes': 15,
                'monitor_window_minutes': '15-20',
                'direction': direction,
                'strength_score': strength,
                'bullish_pct': bullish_pct,
                'bearish_pct': bearish_pct,
                'return_last_15m_pct': round(ret15, 3),
                'return_last_20m_pct': round(ret20, 3),
                'volume_ratio': round(vol_ratio, 2),
                'rsi': round(float(rsi.iloc[-1]), 1),
                'alert': alert,
                'latest_candle_time': str(df.iloc[-1].get('date', 'unknown')),
                'note': 'Heuristic 15-20 minute momentum warning, not a guaranteed forecast. Historical candles are polled; this is not a tick-level WebSocket feed.'}
    except Exception as e:
        raise HTTPException(503, f'Big-move analysis unavailable: {e}')


@app.get('/app', response_class=HTMLResponse)
def dashboard():
    return HTMLResponse('''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>KK NIFTY EARLY SIGNAL</title><style>body{font-family:Arial;background:#101827;color:#eef2ff;margin:0;padding:16px}.wrap{max-width:760px;margin:auto}h2{margin-bottom:4px}.muted{color:#aab6ca;font-size:13px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.card{background:#1d293b;padding:16px;border-radius:14px;margin:12px 0}.big{font-size:30px;font-weight:bold}.green{color:#4ade80}.red{color:#fb7185}button{background:#2563eb;color:white;border:0;border-radius:9px;padding:12px;width:100%;font-size:16px}input{width:100%;box-sizing:border-box;padding:10px;margin:5px 0;border-radius:8px;border:1px solid #475569;background:#0f172a;color:white}.row{display:flex;justify-content:space-between;gap:10px}</style></head><body><div class="wrap"><h2>KK NIFTY EARLY SIGNAL v5</h2><div class="muted">Market analysis dashboard · No auto trading · 15s refresh</div><div class="card"><div class="row"><b>NIFTY 50 LTP</b><span id="ltp">--</span></div><div class="muted" id="time">Waiting for market data</div></div><div class="card"><b>NIFTY 1-Minute Candlestick Chart</b><canvas id="candleChart" style="width:100%;height:260px;display:block"></canvas><div class="muted" id="chartStatus">Waiting for Zerodha candle data</div></div><div class="grid"><div class="card"><b>1 Minute</b><div class="big" id="one">--</div><div id="oneput" class="muted">CALL / PUT</div></div><div class="card"><b>3 Minutes</b><div class="big" id="three">--</div><div id="threeput" class="muted">CALL / PUT</div></div><div class="card"><b>5 Minutes</b><div class="big" id="five">--</div><div id="fiveput" class="muted">CALL / PUT</div></div><div class="card"><b>15 Minutes</b><div class="big" id="fifteen">--</div><div id="fifteenput" class="muted">CALL / PUT</div></div></div><div class="card"><b>15–20 Minute BIG MOVE Early Warning</b><p id="bigmove">Waiting for live market candles.</p><p class="muted">5-second dashboard refresh; underlying candles and Kite data availability determine freshness.</p></div><div class="card"><b>FII / DII Flow</b><p id="fii">Loading NSE FII/DII data...</p><p class="muted" id="fii_trend">Five-day trend loading...</p><button onclick="refreshFii()">Refresh NSE FII / DII</button><p class="muted">NSE daily cash-market data; generally published after market close. Provisional, not live intraday data.</p></div><div class="card"><b>Overnight Gap Forecast</b><p id="gap">Global inputs not entered yet.</p><p class="muted">Gap Up / Gap Down / Flat + estimated gap size</p></div><div class="card"><b>Next Day Outlook</b><p id="next">Awaiting overnight inputs.</p></div><div class="card"><b>Connection</b><p id="conn">Checking Zerodha...</p><button onclick="location.href='/zerodha/login'">Connect / Reconnect Zerodha</button></div><div class="card"><b>Overnight data entry</b><div class="muted">Enter percentage changes from previous close. Leave unknown fields blank.</div><div class="grid"><label>GIFT NIFTY %<input id="gift_nifty_pct" type="number" step="0.01"></label><label>US S&P 500 %<input id="us_sp500_pct" type="number" step="0.01"></label><label>Nasdaq %<input id="us_nasdaq_pct" type="number" step="0.01"></label><label>Asian markets %<input id="asian_markets_pct" type="number" step="0.01"></label><label>Crude change %<input id="crude_pct" type="number" step="0.01"></label><label>USD/INR change %<input id="usd_inr_pct" type="number" step="0.01"></label><label>India VIX change %<input id="india_vix_pct" type="number" step="0.01"></label><label>NIFTY close change %<input id="nifty_close_pct" type="number" step="0.01"></label></div><button onclick="autoInputs()">Fetch automatic global data</button><p class="muted" id="auto_status"></p><button onclick="saveInputs()">Save overnight inputs</button><p class="muted" id="save"></p></div><p class="muted">CALL/PUT values are directional heuristic scores, not calibrated probabilities or guaranteed predictions. Paper-test and use strict risk limits.</p></div><script>async function get(u){let r=await fetch(u);if(!r.ok)throw new Error('HTTP '+r.status+' for '+u);return await r.json()}async function refreshCandles(){const c=document.getElementById('candleChart');const st=document.getElementById('chartStatus');try{const d=await get('/market/candles?interval=minute&days=1');const rows=(d.candles||[]).slice(-80);if(rows.length<2){st.textContent='Not enough candles yet — NO TRADE';return}const w=c.clientWidth||600,h=260,ratio=window.devicePixelRatio||1;c.width=Math.floor(w*ratio);c.height=Math.floor(h*ratio);const x=c.getContext('2d');x.scale(ratio,ratio);x.clearRect(0,0,w,h);const hi=Math.max(...rows.map(a=>Number(a.high))),lo=Math.min(...rows.map(a=>Number(a.low))),pad=(hi-lo)*.08||1,min=lo-pad,max=hi+pad;const y=v=>h-18-((v-min)/(max-min))*(h-36);x.strokeStyle='#334155';x.lineWidth=1;for(let j=0;j<5;j++){const yy=18+j*(h-36)/4;x.beginPath();x.moveTo(0,yy);x.lineTo(w,yy);x.stroke()}const step=w/rows.length,bw=Math.max(2,step*.58);rows.forEach((a,i)=>{const xx=i*step+step/2,o=Number(a.open),cl=Number(a.close),up=cl>=o;x.strokeStyle=up?'#22c55e':'#fb7185';x.fillStyle=up?'#22c55e':'#fb7185';x.beginPath();x.moveTo(xx,y(Number(a.high)));x.lineTo(xx,y(Number(a.low)));x.stroke();const top=Math.min(y(o),y(cl)),bh=Math.max(1,Math.abs(y(o)-y(cl)));x.fillRect(xx-bw/2,top,bw,bh)});const last=rows[rows.length-1];st.textContent='Last candle '+last.date+' · O '+last.open+' H '+last.high+' L '+last.low+' C '+last.close+' · '+rows.length+' candles'}catch(e){st.textContent='Candlestick data unavailable: '+e.message+' — NO TRADE'}}async function refresh(){try{let s=await get('/zerodha/status');document.getElementById('conn').textContent=s.connected?'Connected: '+(s.user_id||'Zerodha'):'Not connected — tap Connect';if(!s.connected){document.getElementById('one').textContent='NO TRADE';document.getElementById('three').textContent='NO TRADE';document.getElementById('five').textContent='NO TRADE';document.getElementById('fifteen').textContent='NO TRADE';document.getElementById('oneput').textContent='Zerodha disconnected';document.getElementById('threeput').textContent='Zerodha disconnected';document.getElementById('fiveput').textContent='Zerodha disconnected';document.getElementById('fifteenput').textContent='Zerodha disconnected'}if(s.connected){let q=await get('/market/nifty/status');document.getElementById('ltp').textContent=q.last_price;document.getElementById('time').textContent=q.timestamp;let f=await get('/market/forecast');for(let [id,x] of [['one',f.one_minute],['three',f.three_minute],['five',f.five_minute],['fifteen',f.fifteen_minute]]){document.getElementById(id).textContent=x.call==null?'NO TRADE':(x.no_trade?'NO TRADE':x.trend+' · '+(x.trend==='BEARISH'?x.put:x.call)+'%');document.getElementById(id+'put').textContent=x.call==null?(x.reason||'Live data unavailable'):('Bullish score '+x.call+'% · Bearish score '+x.put+'% · '+(x.no_trade?'NO ENTRY — WAIT':'WATCH ONLY — confirmation needed')+' · '+x.warning)} }await refreshCandles();let bm=await get('/market/big-move');document.getElementById('bigmove').textContent=bm.available?(bm.alert+' · '+bm.direction+' · Bullish '+bm.bullish_pct+'% · Bearish '+bm.bearish_pct+'% · Move strength '+bm.strength_score+'/100 · 15m move '+bm.return_last_15m_pct+'% · 20m move '+bm.return_last_20m_pct+'% · Vol '+bm.volume_ratio+'x'):bm.message;let fi=await get('/market/institutional');document.getElementById('fii').textContent=fi.available?('Latest '+fi.latest_date+' · FII net ₹'+(fi.daily[0].fii_net_cr??'N/A')+' Cr · DII net ₹'+(fi.daily[0].dii_net_cr??'N/A')+' Cr · '+fi.sentiment_bias):('NSE data unavailable · '+(fi.message||'Try refresh'));document.getElementById('fii_trend').textContent=fi.available?('Available days '+fi.history_days_available+'/5 · 5-day FII ₹'+fi.five_day_fii_net_cr+' Cr · DII ₹'+fi.five_day_dii_net_cr+' Cr · Combined ₹'+fi.five_day_combined_net_cr+' Cr'): (fi.detail||'');let o=await get('/market/overnight');document.getElementById('gap').textContent=o.available?('Gap Up '+o.gap_up+'% · Gap Down '+o.gap_down+'% · Flat '+o.flat+'% · Estimated gap '+o.gap_size_pct+'%'):'Inputs required';let n=await get('/market/next-day');document.getElementById('next').textContent=n.available?('Bullish '+n.bullish+'% · Bearish '+n.bearish+'% · Sideways '+n.sideways+'%'):'Not available until overnight inputs are entered.'}catch(e){document.getElementById('conn').textContent='Data unavailable: '+e.message}}async function refreshFii(){let r=await fetch('/market/institutional-refresh',{method:'POST'});let d=await r.json();document.getElementById('fii').textContent=d.available?('Latest '+d.latest_date+' · FII net ₹'+(d.daily[0].fii_net_cr??'N/A')+' Cr · DII net ₹'+(d.daily[0].dii_net_cr??'N/A')+' Cr · '+d.sentiment_bias):('NSE data unavailable · '+(d.message||''));document.getElementById('fii_trend').textContent=d.available?('Available days '+d.history_days_available+'/5 · 5-day FII ₹'+d.five_day_fii_net_cr+' Cr · DII ₹'+d.five_day_dii_net_cr+' Cr · Combined ₹'+d.five_day_combined_net_cr+' Cr'):(d.detail||'');}async function autoInputs(){let status=document.getElementById('auto_status');status.textContent='Fetching available global-market data...';try{let d=await get('/market/overnight-auto');let names={us_sp500_pct:'us_sp500_pct',us_nasdaq_pct:'us_nasdaq_pct',asian_markets_pct:'asian_markets_pct',crude_pct:'crude_pct',usd_inr_pct:'usd_inr_pct',india_vix_pct:'india_vix_pct',nifty_close_pct:'nifty_close_pct'};for(let [k,id] of Object.entries(names)){if(d.values&&d.values[k]!==undefined)document.getElementById(id).value=d.values[k]}status.textContent=(d.available?'Fetched: '+Object.keys(d.values).join(', '):'No automatic values received')+'. GIFT NIFTY remains blank. '+d.source+' · '+d.fetched_at;refresh()}catch(e){status.textContent='Automatic data unavailable: '+e.message}}async function saveInputs(){let keys=['gift_nifty_pct','us_sp500_pct','us_nasdaq_pct','asian_markets_pct','crude_pct','usd_inr_pct','india_vix_pct','nifty_close_pct'];let d={};for(let k of keys){let v=document.getElementById(k).value;if(v!=='')d[k]=Number(v)}let r=await fetch('/market/overnight-inputs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});document.getElementById('save').textContent=r.ok?'Saved. Forecast updating...':'Could not save inputs';refresh()}refresh();window.addEventListener('resize',refreshCandles);setInterval(refresh,15000);</script></body></html>''')
