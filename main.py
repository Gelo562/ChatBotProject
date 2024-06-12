import spacy
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, CallbackContext
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import random

# Load the spaCy NLP model for Polish
try:
    nlp = spacy.load("pl_core_news_sm")
except OSError as e:
    print(f"Error loading spaCy model: {e}")
    print("Make sure to run: python -m spacy download pl_core_news_sm")
    raise

# Load the sentiment analysis model (trinary)
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# List of diagnostic questions and key words/phrases indicating possible depression symptoms
questions = [
    ("Jak często odczuwasz smutek?", ["często", "codziennie", "zawsze"]),
    ("Czy masz uczucie pustki lub beznadziejności?", ["tak", "zawsze", "często"]),
    ("Czy masz trudności z koncentracją?", ["tak", "często", "zawsze"]),
    ("Czy odczuwasz zmęczenie lub brak energii?", ["tak", "często", "codziennie"]),
    ("Czy masz trudności ze snem?", ["tak", "często", "zawsze"]),
    ("Czy straciłeś zainteresowanie rzeczami, które kiedyś sprawiały Ci przyjemność?", ["tak", "wszystko", "nic"]),
    ("Czy masz problemy z apetytem?", ["tak", "straciłem", "zawsze"]),
    ("Czy masz myśli samobójcze?", ["tak", "często", "zawsze"]),
    ("Czy czujesz się winny lub bezwartościowy?", ["tak", "często", "zawsze"]),
    ("Czy odczuwasz niepokój lub lęk?", ["tak", "często", "zawsze"]),
    ("Czy unikasz kontaktów z innymi ludźmi?", ["tak", "zawsze", "często"]),
    ("Czy masz trudności z podejmowaniem decyzji?", ["tak", "często", "zawsze"]),
    ("Czy masz poczucie, że życie nie ma sensu?", ["tak", "zawsze", "często"]),
    ("Czy odczuwasz frustrację lub irytację?", ["tak", "często", "zawsze"]),
    ("Czy masz poczucie bezradności?", ["tak", "zawsze", "często"]),
    ("Czy masz trudności w pracy lub w szkole?", ["tak", "często", "zawsze"]),
    ("Czy odczuwasz bóle ciała bez wyraźnej przyczyny?", ["tak", "często", "zawsze"]),
    ("Czy masz trudności z utrzymaniem higieny osobistej?", ["tak", "często", "zawsze"]),
    ("Czy odczuwasz obojętność wobec życia?", ["tak", "zawsze", "często"]),
    ("Czy masz trudności z pamięcią?", ["tak", "często", "zawsze"])
]

# Conversation states
QUESTION, ANSWER = range(2)

# Store user responses
user_responses = []

async def start(update: Update, context: CallbackContext) -> int:
    user_responses.clear()
    context.user_data['asked_questions'] = []
    context.user_data['follow_up'] = False  # Flag to track follow-up status
    await update.message.reply_text('Witaj! Jestem chatbotem diagnostycznym. Odpowiedz na kilka pytań, aby pomóc nam ocenić Twój nastrój.')
    await ask_question(update, context)
    return ANSWER

async def ask_question(update: Update, context: CallbackContext):
    context.user_data['follow_up'] = False  # Reset follow-up flag
    available_questions = [q for q, _ in questions if q not in context.user_data['asked_questions']]
    if available_questions:
        question = random.choice(available_questions)
        context.user_data['current_question'] = question
        context.user_data['asked_questions'].append(question)
        await update.message.reply_text(question)
    else:
        diagnosis = diagnose_depression(user_responses)
        await update.message.reply_text(diagnosis)
        return ConversationHandler.END

async def answer(update: Update, context: CallbackContext) -> int:
    user_response = update.message.text
    current_question = context.user_data.get('current_question')

    if not context.user_data['follow_up']:
        is_negative = analyze_response(current_question, user_response)
        user_responses.append((current_question, user_response, is_negative))

    if should_continue_topic(user_response, current_question):
        context.user_data['follow_up'] = True
        await update.message.reply_text("Możesz powiedzieć mi więcej na ten temat?")
    else:
        await ask_question(update, context)
    return ANSWER

def analyze_response(question, response):
    result = sentiment_analyzer(response)[0]
    label = result['label']

    for q, keywords in questions:
        if q == question:
            for keyword in keywords:
                if keyword in response.lower():
                    return True  # Indicates a possible symptom of depression

    return label == 'negative'  # Default to sentiment analysis if no keywords match

def should_continue_topic(user_response, current_question):
    is_negative = analyze_response(current_question, user_response)
    return is_negative

def diagnose_depression(responses):
    total_score = sum(1 for _, _, is_negative in responses if is_negative)
    depression_count = len([1 for _, _, is_negative in responses if is_negative])
    if total_score >= 5:
        return f"Twoje odpowiedzi sugerują, że możesz mieć objawy depresji. Zalecamy konsultację z psychologiem. Liczba odpowiedzi sugerujących depresję: {depression_count}"
    else:
        return f"Twoje odpowiedzi nie sugerują poważnych objawów depresji, ale jeśli masz jakiekolwiek wątpliwości, skonsultuj się z profesjonalistą. Liczba odpowiedzi sugerujących depresję: {depression_count}"

async def cancel(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text('Anulowano rozmowę diagnostyczną. Jeśli potrzebujesz pomocy, skontaktuj się z profesjonalistą.')
    return ConversationHandler.END

def main() -> None:
    application = Application.builder().token("7360025234:AAHzov7nO1jtU0kJJtIV-IV370ocSjAqkyA").build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            ANSWER: [MessageHandler(filters.TEXT & ~filters.COMMAND, answer)]
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    application.add_handler(conv_handler)

    application.run_polling()

if __name__ == '__main__':
    main()
