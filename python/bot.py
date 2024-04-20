import requests
import io
import sys 

#updating about script results, dataframes, graphs, etc

def bot_texter(bot_message, file_data=None, file_name=None, data=None, dataframe=None, name=None):
    
    try:
        bot_token = 'XXXXXXXXXXXXXXXXXXXXX'
        bot_chatID = '123456789'

        if file_data is not None and file_name is not None: #send a picture
            send_file = 'https://api.telegram.org/bot' + bot_token + '/sendDocument?chat_id=' + bot_chatID
            file_bytes = io.BytesIO(file_data)
            response = requests.post(send_file, files={'document': (file_name, file_bytes)})
            
        elif data:
            excel_data = io.BytesIO()
            data.to_excel(excel_data, index=True)  # Export the styled DataFrame to Excel file
            excel_data.seek(0)  # Move the cursor to the beginning of the BytesIO object
            send_file = 'https://api.telegram.org/bot' + bot_token + '/sendDocument?chat_id=' + bot_chatID
            response = requests.post(send_file, files={'document': (f'{name}results.xlsx', excel_data)})

        else: #send a text message
            send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
            response = requests.get(send_text)
            
        return response
    
    except requests.exceptions.RequestException:
        print("Error sending telegram message:", sys.exc_info()[0])
        




def timecalc(start, fin, text, bot=False):
    minutes, seconds = divmod(fin-start, 60)
    hours, minutes = divmod(minutes, 60)
    
    if bot:
        bot_texter("%s: %d:%02d:%02d" % (text, hours, minutes, seconds))
        
    else:
        print("%s: %d:%02d:%02d" % (text, hours, minutes, seconds))
    
    return





def printeo(texto):
    print()
    print(texto)
    print()
    return
