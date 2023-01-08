import requests
import configparser

conf = configparser.ConfigParser()
conf.read('config.ini')
ACCEES_TOKEN = conf['line']['access_token']

# 辞書型を文字列に整形
def pprint(message):
    if isinstance(message, dict):
        s = ''
        for k, v in message.items():
            s += f'{k}:{v}\n'
        message = s
    return '\n'+message

# LINEに通知する
def send_message_to_line(message):
    headers = {
        'Authorization': f'Bearer {ACCEES_TOKEN}'
    }
    data = {'message': pprint(message)}
    requests.post(url='https://notify-api.line.me/api/notify',
                  headers=headers,
                  data=data)

if __name__ == '__main__':
    send_message_to_line('test')
    send_message_to_line({'last': 100, 'ask': 99, 'volume': 1000})