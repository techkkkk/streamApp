import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
# import plotly.graph_objs as go
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from streamlit_autorefresh import st_autorefresh
from concurrent.futures import ThreadPoolExecutor # 导入线程池

# ==============================================================================
# 1. 日志与页面配置
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
st.set_page_config(layout="wide", page_title="加密货币监控")
st.title("🚀 加密货币监控仪表盘")

# ==============================================================================
# 2. Session State & URL 管理
# ==============================================================================
# 初始化会话状态
if 'last_notify' not in st.session_state:
    st.session_state.last_notify = {}
# 新增：初始化信号历史记录
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []

def get_tasks_from_params():
    tasks_str = st.query_params.get("tasks", "")
    if not tasks_str: return []
    return [{"id": t, "symbol": t.split('-')[0], "interval": t.split('-')[1]} for t in tasks_str.split(',') if '-' in t]

def set_tasks_to_params(tasks):
    if not tasks: st.query_params.clear()
    else: st.query_params["tasks"] = ",".join([task['id'] for task in tasks])

# ==============================================================================
# 3. 核心数据与通知函数
# ==============================================================================
@st.cache_data(ttl=3600)
def get_all_symbols():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        logger.info("Fetching symbol list...")
        response = requests.get(url, timeout=10); response.raise_for_status()
        data = response.json()
        symbols = sorted([s["symbol"] for s in data["symbols"] if s["status"] == "TRADING" and "USDT" in s["symbol"]])
        logger.info(f"Fetched {len(symbols)} USDT symbols.")
        return symbols
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch symbol list: {e}"); st.error(f"无法获取交易对列表: {e}"); return []

def format_price(price: float) -> str:
    if price >= 10: return f"{price:,.2f}"
    elif price >= 1: return f"{price:.4f}"
    else: return f"{price:.8f}"

@st.cache_data(ttl=10)
def get_analyzed_data(symbol: str, interval: str):
    logger.info(f"Thread for {symbol}-{interval} started.")
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=500"
        data = requests.get(url, timeout=10).json()
        df = pd.DataFrame(data, columns=["open_time","open","high","low","close","volume","close_time","quote_asset_volume","trades","taker_buy_base","taker_buy_quote","ignore"])
        for col in ["open", "high", "low", "close", "volume"]: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.index = pd.to_datetime(df['close_time'], unit='ms')

        delta = df['close'].diff(); gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14, 1).mean(); avg_loss = loss.rolling(14, 1).mean()
        df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss).replace([np.inf, -np.inf], np.nan).ffill()))
        low_min = df['low'].rolling(9, 1).min(); high_max = df['high'].rolling(9, 1).max()
        rsv = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['K'] = rsv.replace([np.inf, -np.inf], np.nan).ffill().ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean(); df['J'] = 3 * df['K'] - 2 * df['D']

        signal_text = "—"
        if len(df) > 1:
            i, prev_i = -1, -2
            if df['RSI'].iloc[prev_i] > 70 and df['RSI'].iloc[i] <= 70 and df['K'].iloc[prev_i] > df['D'].iloc[prev_i] and df['K'].iloc[i] <= df['D'].iloc[i]:
                signal_text = "⬇️ 死叉卖出"
            elif df['RSI'].iloc[prev_i] < 30 and df['RSI'].iloc[i] >= 30 and df['K'].iloc[prev_i] < df['D'].iloc[prev_i] and df['K'].iloc[i] >= df['D'].iloc[i]:
                signal_text = "⬆️ 金叉买入"
        
        logger.info(f"Thread for {symbol}-{interval} finished successfully.")
        return {"ok": True, "df": df, "latest": df.iloc[-1], "signal": signal_text}
    except Exception as e:
        logger.error(f"Thread for {symbol}-{interval} failed: {e}")
        return {"ok": False, "error": f"处理错误"}

def send_telegram_message(bot_token, chat_id, message):
    try: requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage", data={"chat_id": chat_id, "text": message}, timeout=5)
    except Exception as e: logger.error(f"Failed to send Telegram message: {e}")

def send_email(subject, body, smtp_host, smtp_port, smtp_user, smtp_pass, email_to_list):
    msg = MIMEMultipart(); msg['From'] = smtp_user; msg['To'] = ", ".join(email_to_list); msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls(); server.login(smtp_user, smtp_pass); server.send_message(msg)
    except Exception as e: logger.error(f"Failed to send email: {e}")

def plot_task(df, task_id):
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
    fig.update_layout(title=f"{task_id} K线图", xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=40, b=20), height=400)
    return fig

# ==============================================================================
# 4. 侧边栏配置
# ==============================================================================
with st.sidebar:
    st.header("⚙️ 全局配置")
    refresh_sec = st.number_input("数据刷新间隔(秒)", 10, 3600, 60)
    notify_interval = st.number_input("同类信号通知最小间隔(秒)", 60, 86400, 300)
    with st.expander("📢 Telegram 配置"):
        telegram_notify = st.checkbox("启用 Telegram 通知"); bot_token = st.text_input("Bot Token", type="password"); chat_id = st.text_input("Chat ID")
    with st.expander("📧 邮件通知配置"):
        email_notify = st.checkbox("启用邮件通知"); smtp_host = st.text_input("SMTP Host", "smtp.example.com"); smtp_port = st.number_input("SMTP Port", 587); smtp_user = st.text_input("发件邮箱"); smtp_pass = st.text_input("邮箱密码", type="password"); email_to = st.text_area("收件人邮箱（逗号分隔）")

# ==============================================================================
# 5. 主应用流程
# ==============================================================================
st_autorefresh(interval=refresh_sec * 1000, key="data_refresher")

current_tasks = get_tasks_from_params()
all_symbols = get_all_symbols()

with st.container(border=True):
    st.subheader("➕ 任务管理")
    if all_symbols:
        default_symbols = [task['symbol'] for task in current_tasks]; all_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        current_interval = current_tasks[0]['interval'] if current_tasks else '1h'
        col1, col2, col3 = st.columns([2, 1, 1]); selected_symbols = col1.multiselect("选择交易对", all_symbols, default_symbols)
        interval_to_set = col2.selectbox("选择周期", all_intervals, index=all_intervals.index(current_interval))
        col3.write("");
        if col3.button("🚀 更新任务列表", use_container_width=True, type="primary"):
            new_tasks = [{"id": f"{s}-{interval_to_set}", "symbol": s, "interval": interval_to_set} for s in selected_symbols]
            set_tasks_to_params(new_tasks); st.rerun()

st.divider()

# 修改：增加一个“信号历史”选项卡
tab1, tab2, tab3 = st.tabs(["📊 实时总览", "📈 K线图详情", "📜 信号历史"])

with tab1:
    if not current_tasks:
        st.info("当前没有监控任务，请在上方添加。")
    else:
        results = []
        with st.spinner(f"正在并行刷新 {len(current_tasks)} 个任务的数据..."):
            with ThreadPoolExecutor(max_workers=10) as executor:
                args_to_pass = [(task['symbol'], task['interval']) for task in current_tasks]
                results = list(executor.map(lambda p: get_analyzed_data(*p), args_to_pass))
        
        dashboard_rows = []
        for task, result in zip(current_tasks, results):
            if result["ok"]:
                latest = result["latest"]
                dashboard_rows.append({
                    "任务ID": task['id'], "价格": format_price(latest['close']), "RSI": f"{latest['RSI']:.2f}",
                    "K": f"{latest['K']:.2f}", "D": f"{latest['D']:.2f}", "J": f"{latest['J']:.2f}", "信号": result["signal"]
                })
                # --- 通知与历史记录逻辑 ---
                if "—" not in result["signal"]:
                    last_time = st.session_state.last_notify.get(task['id'], {}).get(result["signal"], 0)
                    if time.time() - last_time > notify_interval:
                        trigger_time = latest.name.strftime('%Y-%m-%d %H:%M:%S')
                        message = f"📈 **加密货币信号**\n\n**交易对**: {task['id']}\n**信号类型**: {result['signal']}\n**时间**: {trigger_time}\n**价格**: {format_price(latest['close'])}"
                        if telegram_notify and bot_token and chat_id: send_telegram_message(bot_token, chat_id, message)
                        if email_notify and smtp_user and smtp_pass and email_to:
                            recipients = [e.strip() for e in email_to.split(",") if e.strip()]
                            if recipients: send_email(f"信号提醒: {task['id']}", message, smtp_host, smtp_port, smtp_user, smtp_pass, recipients)
                        st.toast(f"已发送 {task['id']} 的 {result['signal']} 通知！", icon="📬")
                        st.session_state.last_notify.setdefault(task['id'], {})[result["signal"]] = time.time()
                        
                        # 新增：将信号添加到历史记录
                        st.session_state.signal_history.insert(0, {
                            "时间": trigger_time,
                            "交易对": task['symbol'],
                            "K线级别": task['interval'],
                            "信号类型": result['signal'],
                            "触发价格": format_price(latest['close'])
                        })
            else:
                dashboard_rows.append({"任务ID": task['id'], "价格": "加载失败", "信号": result.get("error")})
        
        if dashboard_rows:
            st.dataframe(pd.DataFrame(dashboard_rows).set_index("任务ID"), use_container_width=True)

with tab2:
    if not current_tasks:
        st.info("添加任务后，可在此处查看详细K线图。")
    else:
        for task in current_tasks:
            with st.expander(f"查看 {task['id']} 的K线图"):
                result = get_analyzed_data(task["symbol"], task["interval"])
                if result and result["ok"]:
                    st.plotly_chart(plot_task(result["df"], task['id']), use_container_width=True)
                else:
                    st.error(f"无法为 {task['id']} 加载图表。原因: {result.get('error', '未知') if result else '未加载'}")

# 新增：信号历史面板
with tab3:
    st.subheader("📜 最近触发的信号历史")
    if not st.session_state.signal_history:
        st.info("目前还没有任何信号被触发。")
    else:
        # 创建一个按钮来清空历史记录
        if st.button("🗑️ 清空历史记录"):
            st.session_state.signal_history = []
            st.rerun()

        # 将历史记录转换为DataFrame并显示
        history_df = pd.DataFrame(st.session_state.signal_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)