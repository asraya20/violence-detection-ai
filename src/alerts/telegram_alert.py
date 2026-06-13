import requests

BOT_TOKEN = "7910797921:AAEbUnZsOw6KguUU02lPHX9_6EYFK52x0MA"
CHAT_ID = "5053300092"

def send_telegram_alert(message):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    requests.post(url, data=payload)