
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="DESSAINT Studio | Customer Support AI",
    page_icon="🛍️",
    layout="centered",
)

INTENTS = [
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "check_cancellation_fee",
    "check_invoice",
    "check_payment_methods",
    "check_refund_policy",
    "complaint",
    "contact_customer_service",
    "contact_human_agent",
    "create_account",
    "delete_account",
    "delivery_options",
    "delivery_period",
    "edit_account",
    "get_invoice",
    "get_refund",
    "newsletter_subscription",
    "payment_issue",
    "place_order",
    "recover_password",
    "registration_problems",
    "review",
    "set_up_shipping_address",
    "switch_account",
    "track_order",
    "track_refund",
]

INTENT_MODEL = "davidgv92/dessaint-distilbert-intent-classifier"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


@st.cache_resource
def load_models():
    intent_classifier = pipeline(
        "text-classification",
        model=INTENT_MODEL,
        device=-1,
    )

    sentiment_classifier = pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        device=-1,
    )

    return intent_classifier, sentiment_classifier


st.title("DESSAINT Studio")
st.subheader("AI-Powered Customer Support Classification")

st.write(
    "Analyze customer support messages by identifying their "
    "intent and sentiment. This is a prototype and does not "
    "connect to DESSAINT Studio's internal systems."
)

message = st.text_area(
    "Customer support message",
    placeholder="Example: Where is my order? It was supposed to arrive yesterday.",
    height=130,
)

if st.button("Analyze message", type="primary"):
    if not message.strip():
        st.warning("Please enter a customer support message.")
    else:
        try:
            with st.spinner("Analyzing message..."):
                intent_classifier, sentiment_classifier = load_models()

                intent_result = intent_classifier(message)[0]
                sentiment_result = sentiment_classifier(message)[0]

            predicted_label = intent_result["label"]

            if predicted_label.startswith("LABEL_"):
                intent_index = int(predicted_label.split("_")[1])
                predicted_intent = INTENTS[intent_index]
            else:
                predicted_intent = predicted_label

            st.divider()
            st.subheader("Analysis results")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Predicted intent**")
                st.write(predicted_intent.replace("_", " ").title())
                st.metric(
                    "Intent confidence",
                    f"{intent_result['score']:.1%}",
                )

            with col2:
                st.markdown("**Customer sentiment**")
                st.write(sentiment_result["label"].title())
                st.metric(
                    "Sentiment confidence",
                    f"{sentiment_result['score']:.1%}",
                )

            st.caption(
                "Predictions are model estimates and may be incorrect."
            )

        except Exception as error:
            st.error(
                "The models could not be loaded or the message "
                "could not be analyzed. Please try again."
            )
            st.exception(error)
