import spacy
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, CallbackContext
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, Conversation, ConversationalPipeline
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

import spacy
import os
import re

#################################################################
##### ETAP 1 -> Przygotowanie artykułów i ich analiza ###########
#################################################################

#deklarowanie NLP - modelu do przetwarzania języka naturalnego
#wykorzystano model pl_core_news_sm z bibliotaki spaCy
nlp = spacy.load("pl_core_news_sm")

#funkcja wczytująca zewnętrzne pliki (artykuły) do dalszej analizy
def load_texts(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

#funkcja wykorzystująca NLP do wyciągnięcia z artykułów symptomów i oznak
#wskazujących na depresję. Funkcja przetwarza artykuły uwzględniając
#słowa kluczowe w zdaniach
def extract_symptoms(texts, symptom_keywords):
    symptoms = []
    for text in texts:
        doc = nlp(text)
        for sent in doc.sents:
            if any(keyword in sent.text.lower() for keyword in symptom_keywords):
                symptoms.append(sent.text)
    return symptoms

#zdefiniowanie zmiennych jako lokalizacji artykułów oraz słów kluczowych do wyszukiwania symptomów / oznak
directory = 'articles/'
symptom_keywords = [
    'smutek', 'apatia', 'brak energii', 'utrata zainteresowań', 'problemy ze snem',
    'zmiany apetytu', 'niskie poczucie własnej wartości', 'poczucie winy',
    'problemy z koncentracją', 'myśli samobójcze', 'ból',
    'uczucie pustki', 'trudności z podejmowaniem decyzji', 'uczucie przytłoczenia',
    'zmniejszona wydajność', 'trudności z pamięcią', 'unikanie kontaktów społecznych',
    'negatywne myśli o przyszłości', 'nadmierne zamartwianie się', 'poczucie osamotnienia',
    'utrata zainteresowania w codziennych czynnościach', 'płaczliwość', 'trudności w relacjach',
    'zmniejszona samoocena', 'problemy z motywacją', 'samookaleczanie się'
]


#wywołanie funkcji wczytującej artykuły
texts = load_texts(directory)

#wywołanie funkcji wyciągającej potrzebne dane z artykułów
symptoms = extract_symptoms(texts, symptom_keywords)

#scalanie rezultatu
result = ""
for symptom in symptoms:
    result = result + symptom



#################################################################
###### ETAP 2 -> Segregacja objawów i przygotowanie pytań #######
#################################################################

#funkcja klasyfikująca symptomy do odpowiednich kategorii na podstawie ich przesłania
#dalej w kodzie będzie to wykorzystywane do nadania pytaniom odpowieniego kierunku, na przykład:
#Czy obserwujesz u siebie objawy, Czy cierpisz na przypadłość...
#funkcja sprawdzi każdą linię z rezultatów - jeśli kończy się znakiem :, linia uznawana jest za kategorię,
#w przeciwnym przypadku, za symptom
def segregate_symptoms(results):
    categories = {}
    current_category = None
    for line in results.split('\n'):
        if not line:
            continue
        if re.match(r'.+:$', line):
            current_category = line
            categories[current_category] = []
        elif current_category:
            categories[current_category].append(line)
    return categories

#wywołanie funkcji segregującej symptomy
data = segregate_symptoms(result)

#zdefiniowanie tablic dla odpowiendich kategorii pytań
symptoms_diseases = []
symptoms_symptoms = []
symptoms_states = []
symptoms_environment = []

#metoda czyszcząca symptomy, zapobiegająca sytuacji wystąpienia w pytaniu podwójnego znaku na końcu zdania
def clean_symptom(symptom):
    if symptom[-1] in [';', ',', '.']:
        return symptom[:-1]
    return symptom

#pętla przyporządkowująca symptomy do odpowiednich tablic na podstawie nazwy kategorii
#i potencjalnie występujących w niej słów
for category, symptoms in data.items():
    if "nne choroby" in category:
        for symptom in symptoms:
            if symptom[-1] != '?':
                symptoms_diseases.append(clean_symptom(symptom))
    elif "towarzysz" in category or "wpływa" in category or "objaw" in category:
        for symptom in symptoms:
            if symptom[-1] != '?' and ' to ' not in symptom:
                symptoms_symptoms.append(clean_symptom(symptom))
    elif "stan" in category or "nastr" in category or "okres" in category:
        for symptom in symptoms:
            if symptom[-1] != '?':
                symptoms_states.append(clean_symptom(symptom))
    elif "otoczeni" in category:
        for symptom in symptoms:
            if symptom[-1] != '?':
                symptoms_environment.append(clean_symptom(symptom))

#zadeklarowanie pustej tablicy, która dalej w kodzie będzie zawierała pytania wykorzystywane przez chatBota
questions = []

#funkcja formująca odpowiednio pytanie na podstawie typu symptomu oraz określająca
#potencjalne odpowiedzi uzytkownika wskazujące na rozpoznanie choroby
def form_questions(symptoms, type):
    if (type == "diseases"):
        question_Form = "Czy cierpisz na przypadłość o nazwie "
        answers = ["tak", "cierpię", "choruję", "mam"]
    elif (type == "symptoms"):
        question_Form = "Czy obserwujesz u siebie objawy: "
        answers = ["tak", "obserwuję", "często", "czasem", "trochę","ciągle"]
    elif (type == "states"):
        question_Form = "Czy miewasz stan: "
        answers = ["tak", "miewam", "często", "zdarza się", "czasami", "ciągle"]
    elif (type == "environment"):
        question_Form = "Czy obserwujesz u siebie zjawisko:  "
        answers = ["tak", "obserwuję", "zdarza się", "czasami", "często", "trochę","ciągle"]
    for symptom in symptoms:
        question = question_Form + symptom + '?'
        final_question = (question, answers)
        questions.append(final_question)

#wywołanie funkcji formującej pytania dla każdej z tablic
form_questions(symptoms_diseases, "diseases")
form_questions(symptoms_symptoms, "symptoms")
form_questions(symptoms_states, "states")
form_questions(symptoms_environment, "environment")

# for question_one in questions:
#     print(question_one)

#################################################################
################ ETAP 3 -> Weryfikacja wyników ##################
#################################################################

#funkcja walidująca konkretne pytanie pod kątem jego długości i poprawności
#wykorzystująca do tego NLP. Poprawność weryfikowana jest na podstawie konstrukcji zdania
#w oparciu o występowanie rzeczownika
def is_valid_question(question):
    if len(question.split()) > 20:
        return False
    doc = nlp(question)
    has_noun = any(token.pos_ == "NOUN" for token in doc)
    return has_noun

#funkcja wykorzystująca powyższy kod do wywołania walidacji na każdym z pytań
#poprawne pytania dodawane są do tablicy, która następnie zastąpi obecne question
def filter_questions(questions):
    valid_questions = []
    for question, answers in questions:
        if is_valid_question(question) and (question, answers) not in valid_questions:
            valid_questions.append((question, answers))
    return valid_questions

#wywołanie metody walidującej
questions = filter_questions(questions)

#wyświetlenie każdego z poprawnych pytań
for question in questions:
    print(question)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Load the spaCy NLP model for Polish
try:
    nlp = spacy.load("pl_core_news_sm")
except OSError as e:
    print(f"Error loading spaCy model: {e}")
    print("Make sure to run: python -m spacy download pl_core_news_sm")
    raise

# Load the sentiment analysis model (trinary)
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer_sentiment = AutoTokenizer.from_pretrained(model_name)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=model_sentiment, tokenizer=tokenizer_sentiment)

# Load the text generation model
model_name_causal = "microsoft/DialoGPT-medium"
tokenizer_causal = AutoTokenizer.from_pretrained(model_name_causal, use_fast=True)
tokenizer_causal.pad_token = tokenizer_causal.eos_token

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Set torch_dtype based on CUDA availability
if cuda_available and torch.cuda.is_bf16_supported():
    torch_dtype = torch.bfloat16
elif cuda_available:
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

model_causal = AutoModelForCausalLM.from_pretrained(
    model_name_causal,
    torch_dtype=torch_dtype,
    device_map="auto" if cuda_available else None
)

# Conversation states
QUESTION, ANSWER = range(2)

# Store user responses
user_responses = []

async def start(update: Update, context: CallbackContext) -> int:
    user_responses.clear()
    context.user_data['asked_questions'] = []
    context.user_data['question_count'] = 0  # licznik pytań
    context.user_data['follow_up'] = False  # Flag to track follow-up status
    await update.message.reply_text('Witaj! Jestem chatbotem diagnostycznym. Odpowiedz na kilka pytań, aby pomóc nam ocenić Twój nastrój.')
    await ask_question(update, context)
    return ANSWER


async def ask_question(update: Update, context: CallbackContext):
    context.user_data['follow_up'] = False  # Reset follow-up flag

    # Sprawdź, czy zadano już 10 pytań
    if context.user_data['question_count'] >= 10:
        diagnosis = diagnose_depression(user_responses)
        await update.message.reply_text(diagnosis)
        return ConversationHandler.END

    available_questions = [q for q, _ in questions if q not in context.user_data['asked_questions']]
    if available_questions:
        question = random.choice(available_questions)
        context.user_data['current_question'] = question
        context.user_data['asked_questions'].append(question)
        context.user_data['question_count'] += 1  # Zwiększ licznik pytań
        await update.message.reply_text(question)
    else:
        diagnosis = diagnose_depression(user_responses)
        await update.message.reply_text(diagnosis)
        return ConversationHandler.END

async def answer(update: Update, context: CallbackContext) -> int:
    # Sprawdź, czy wiadomość zawiera zdjęcie
    if update.message.photo:
        await update.message.reply_text("Otrzymałem zdjęcie. Proszę odpowiedzieć na pytanie tekstowo.")
        return ANSWER
    
    user_response = update.message.text
    current_question = context.user_data.get('current_question')
    
    # Jeśli nie jest to wiadomość follow-up
    if not context.user_data['follow_up']:
        is_negative = analyze_response(current_question, user_response)
        user_responses.append((current_question, user_response, is_negative))

    # Jeśli odpowiedź jest negatywna, kontynuuj temat
    if should_continue_topic(user_response, current_question):
        context.user_data['follow_up'] = True
        follow_up_question = await generate_follow_up(current_question, user_response)
        await update.message.reply_text(follow_up_question)
    else:
        await ask_question(update, context)
    return ANSWER


#Analizuje odpowiedź użytkownika, sprawdzając, czy jest negatywna
def analyze_response(question, response):
    result = sentiment_analyzer(response)[0]
    label = result['label']
    
    # Mapping sentiment labels to determine if the response is negative
    if label == 'negative':
        return True

    for q, keywords in questions:
        if q == question:
            for keyword in keywords:
                if keyword in response.lower():
                    return True  # Indicates a possible symptom of depression
    
    return False  # Default to False if no keywords match and sentiment is not negative


#Sprawdza, czy należy kontynuować temat na podstawie analizy odpowiedzi użytkownika
def should_continue_topic(user_response, current_question):
    is_negative = analyze_response(current_question, user_response)
    return is_negative


#Ekstrahuje pytanie follow-up z wygenerowanego tekstu
def extract_follow_up(text):
    question_pos = text.find('?')
    if question_pos != -1:
        return text[:question_pos + 1]
    else:
        return None

#Generuje pytanie follow-up, jeśli odpowiedź użytkownika jest negatywna
async def generate_follow_up(question, user_response, max_attempts=5):
    chat = [
        {"role": "system",
         "content": "Jesteś terapeutą diagnozującym depresję. Okaż zrozumienie dla użytkownika i dopytaj o jego problem w kulturalny i delikatny"},
        {"role": "assistant", "content": question},
        {"role": "user", "content": user_response}
    ]
    inputs = tokenizer_causal.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")

    for attempt in range(max_attempts):
        with torch.no_grad():
            outputs = model_causal.generate(
                inputs,
                pad_token_id=tokenizer_causal.eos_token_id,
                max_new_tokens=128,
                temperature=0.2,
                repetition_penalty=1.15,
                top_p=0.95,
                do_sample=True
            )

        new_tokens = outputs[0, inputs.size(1):]
        response = tokenizer_causal.decode(new_tokens, skip_special_tokens=True)
        follow_up_question = extract_follow_up(response)
        if follow_up_question:
            return follow_up_question

    # Fallback if no valid follow-up question found after max_attempts
    fallback_question = "Czy możesz powiedzieć mi więcej na ten temat?"
    return fallback_question

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
    application = Application.builder().token("7360025234:AAHzov7nO1jtU0kJJtIV-IV370ocSjAqkyA").build() #7479723528:AAGTmj-KwKhdhTByObJZtqvVaO0_nL1QI6I
    #7360025234:AAHzov7nO1jtU0kJJtIV-IV370ocSjAqkyA

    conv_handler = ConversationHandler(
    entry_points=[CommandHandler('start', start)],
    states={
        ANSWER: [MessageHandler(filters.TEXT & ~filters.COMMAND, answer),
                 MessageHandler(filters.PHOTO, answer)]
    },
    fallbacks=[CommandHandler('cancel', cancel)],
)

    application.add_handler(conv_handler)

    application.run_polling()

if __name__ == '__main__':
    main()
