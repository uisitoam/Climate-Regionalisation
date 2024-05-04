import requests
import io
import sys 





#updating about script results, graphs, etc
def bot_texter(bot_message, file_data=None, file_name=None):
    """
    This function sends a message through a telegram bot.

    Parameters
    ----------
    bot_message : str
        The message to be sent by the bot.
    file_data : bytes, optional
        The bytes of the file to be sent by the bot.
    file_name : str, optional
        The name of the file to be sent by the bot.

    Returns
    -------
    requests.models.Response
        The response object from the Telegram API.
    
    Raises
    ------
    requests.exceptions.RequestException
        An error occurred while sending the message or file.
    
    Notes
    -----
    - The function uses the Telegram Bot API to send messages and files to a specified chat.
    - The bot token and chat ID are hard-coded in the function and should be updated as needed.
    - The function can send either a text message or a file, depending on the input parameters.

    """
    
    try:
        bot_token = 'XXXXXXXXXXXXX'
        bot_chatID = '123456789'

        if file_data is not None and file_name is not None: #send a picture
            send_file = 'https://api.telegram.org/bot' + bot_token + '/sendDocument?chat_id=' + bot_chatID
            file_bytes = io.BytesIO(file_data)
            response = requests.post(send_file, files={'document': (file_name, file_bytes)})

        else: #send a text message
            send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
            response = requests.get(send_text)
            
        return response
    
    except requests.exceptions.RequestException:
        print("Error sending telegram message:", sys.exc_info()[0])
        




def timecalc(start, fin, text, bot=False):
    """
    This function calculates the time taken for a process to complete and prints the result
    in HH:MM:SS format.

    Parameters
    ----------
    start : float
        The starting time of the process.
    fin : float
        The ending time of the process.
    text : str
        The text to be displayed along with the time taken.
    bot : bool, optional
        A flag indicating whether the message should be sent by the bot. The default is False.

    Returns
    -------
    None
        This function does not return any value. It prints the time taken for the process.

    Raises
    ------
    None
    
    Notes
    -----
    - The function calculates the time taken for a process to complete in seconds.
    - The function converts the time taken to HH:MM:SS format.
    - The function prints the text along with the time taken.
    - The function can send the message through a Telegram bot if the bot flag is set to True.
    
    """
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
