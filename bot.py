import logging
import asyncio
import torch
import os
import aiogram.utils.markdown as fmt
from aiogram import Bot, Dispatcher, executor, types
from aiogram.utils.exceptions import BotBlocked
from os import getenv
from sys import exit
# Data processing part
from text_processing.text_data_extraction import GoemotionsAnalyzerBasic, GoEmotionsAnalyzerGroup, BertClassifier,\
     classify_custom

# Initialize NN
sentiment_analyzer = GoEmotionsAnalyzerGroup()
emotion_range_analyzer = GoemotionsAnalyzerBasic()
# Custom classifier
dirname = os.path.dirname(__file__)
weights_path = os.path.join(dirname, 'text_processing/text_processing_data/CEM_classifier_model_v1.pt')

custom_classifier = BertClassifier()
custom_classifier.load_state_dict(torch.load(weights_path))

print("Models initialized.")

# Bot object
# Generate bot token with BotFather
# Token was moved to env variable
bot_token = getenv("BOT_TOKEN")
if not bot_token:
    exit("Error: no token provided")

bot = Bot(token=bot_token)
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO)
logging.log(level=10, msg="Base setup complete!")


@dp.message_handler(commands="block")
async def cmd_block(message: types.Message):
    await asyncio.sleep(10.0)  # asyncio sleep
    await message.reply("You have been blocked!")


@dp.errors_handler(exception=BotBlocked)
async def error_bot_blocked(update: types.Update, exception: BotBlocked):
    # Update: Telegram event object. Exception: exception object
    # And we can somehow process this user - delete him from DB, etc
    print(f"The user has blocked me.\nMessage: {update}\nError: {exception}")
    return True


# @dp.message_handler(commands="test4")
# async def with_hidden_link(message: types.Message):
#     await message.answer(
#         f"{fmt.hide_link('https://skylum.com/luminar-ai-b')}"
#         f"The first image editor fully powered by artificial intelligence.",
#         parse_mode=types.ParseMode.HTML)

@dp.message_handler()
async def echo(message: types.Message):
    message_txt = message.text
    try:
        sentiment = sentiment_analyzer.goemotions(message_txt[:512])[0]['labels'][0]
    except IndexError as e:
        print(e)
        sentiment = [{'label': 'N/A', 'score': 'N/A'}]
    emotions = emotion_range_analyzer.goemotions(message_txt[:512])[0]['labels']
    ticket_class = classify_custom(custom_classifier, message_txt)

    output_str = f"{ticket_class}Message sentiment: {sentiment}\nMessage emotions: {emotions}"
    # await message.answer(emotions_result)
    await message.reply(output_str)


if __name__ == "__main__":
    # Launch bot
    executor.start_polling(dp, skip_updates=True)
