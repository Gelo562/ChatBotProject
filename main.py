import spacy
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, CallbackContext
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import random

# Ładowanie modelu NLP spaCy
nlp = spacy.load("pl_core_news_sm")

# Ładowanie modelu herBERT do analizy sentymentu
model_name = "allegro/herbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Lista pytań diagnostycznych i kluczowych słów/fraz wskazujących na możliwy objaw depresji
questions = [
    ("Jak często odczuwasz smutek?", ["często", "codziennie", "zawsze", "od czasu do czasu"]),
    ("Czy masz uczucie pustki lub beznadziejności?", ["tak", "zawsze", "często", "czasami"]),
    ("Czy masz trudności z koncentracją?", ["tak", "często", "zawsze"]),
    ("Czy odczuwasz zmęczenie lub brak energii?", ["tak", "często", "codziennie"]),
    ("Czy masz trudności ze snem?", ["tak", "często", "zawsze"]),
    ("Czy straciłeś zainteresowanie rzeczami, które kiedyś sprawiały Ci przyjemność?", ["tak", "wszystko", "nic"]),
    ("Czy masz problemy z apetytem?", ["tak", "straciłem", "zawsze"]),
    ("Czy masz myśli samobójcze?", ["tak", "często", "zawsze"]),
    ("Czy czujesz się winny lub bezwartościowy?", ["tak", "często", "zawsze"])
]

# Stany konwersacji
QUESTION, ANSWER = range(2)

# Przechowywanie odpowiedzi użytkownika
user_responses = []

async def start(update: Update, context: CallbackContext) -> int:
    user_responses.clear()
    context.user_data['asked_questions'] = []
    await update.message.reply_text('Witaj! Jestem chatbotem diagnostycznym. Odpowiedz na kilka pytań, aby pomóc nam ocenić Twój nastrój.')
    await ask_question(update, context)
    return ANSWER

async def ask_question(update: Update, context: CallbackContext):
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
    is_negative = analyze_response(current_question, user_response)
    user_responses.append((current_question, user_response, is_negative))
    
    if should_continue_topic(user_response, current_question):
        await update.message.reply_text("Możesz powiedzieć mi więcej na ten temat?")
    else:
        await ask_question(update, context)
    return ANSWER

def analyze_response(question, response):
    # Użycie modelu herBERT do analizy sentymentu odpowiedzi
    result = sentiment_analyzer(response)[0]
    sentiment_score = 1 if result['label'] == 'LABEL_1' else 0

    # Sprawdzanie kluczowych słów/fraz
    for q, keywords in questions:
        if q == question:
            for keyword in keywords:
                if keyword in response.lower():
                    return True  # Odpowiedź wskazuje na możliwy objaw depresji

    return sentiment_score > 0  # Domyślnie używamy analizy sentymentu, jeśli brak kluczowych słów

def should_continue_topic(user_response, current_question):
    # Logika decydująca, czy kontynuować wątek na podstawie analizy sentymentu i treści odpowiedzi użytkownika
    is_negative = analyze_response(current_question, user_response)
    print(is_negative)
    return is_negative  # Kontynuuj wątek, jeśli odpowiedź była negatywna

def diagnose_depression(responses):
    total_score = sum(1 for _, _, is_negative in responses if is_negative)
    if total_score >= 5:
        return "Twoje odpowiedzi sugerują, że możesz mieć objawy depresji. Zalecamy konsultację z psychologiem."
    else:
        return "Twoje odpowiedzi nie sugerują poważnych objawów depresji, ale jeśli masz jakiekolwiek wątpliwości, skonsultuj się z profesjonalistą."

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
