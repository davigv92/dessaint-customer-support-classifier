
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


st.divider()
st.header("Evaluate model on labeled data")

st.write(
    "Upload a CSV containing an `instruction` column with customer "
    "messages and an `intent` column with the correct intent labels. "
    "The app will compare predictions with the supplied labels."
)

uploaded_file = st.file_uploader(
    "Upload evaluation CSV",
    type=["csv"],
    key="evaluation_csv",
)

if uploaded_file is not None:
    import pandas as pd
    from sklearn.metrics import accuracy_score

    evaluation_df = pd.read_csv(uploaded_file)

    required_columns = {"instruction", "intent"}

    if not required_columns.issubset(evaluation_df.columns):
        st.error("The CSV must contain 'instruction' and 'intent' columns.")
    elif evaluation_df.empty:
        st.error("The uploaded CSV contains no examples.")
    elif evaluation_df["instruction"].isna().any() or evaluation_df["intent"].isna().any():
        st.error("The CSV contains missing messages or labels.")
    else:
        if st.button("Run evaluation", type="primary"):
            try:
                intent_classifier, _ = load_models()

                with st.spinner("Evaluating messages..."):
                    messages = evaluation_df["instruction"].astype(str).tolist()
                    true_labels = evaluation_df["intent"].astype(str).tolist()

                    predictions = intent_classifier(
                        messages,
                        batch_size=16,
                        truncation=True,
                    )

                    predicted_labels = []
                    for prediction in predictions:
                        label = prediction["label"]

                        if label.startswith("LABEL_"):
                            index = int(label.split("_")[1])
                            predicted_labels.append(INTENTS[index])
                        else:
                            predicted_labels.append(label)

                accuracy = accuracy_score(true_labels, predicted_labels)
                results_df = evaluation_df.copy()
                results_df["predicted_intent"] = predicted_labels
                results_df["correct"] = (
                    results_df["intent"].astype(str)
                    == results_df["predicted_intent"]
                )

                correct_count = int(results_df["correct"].sum())

                st.metric("Evaluation sample size", len(results_df))
                st.metric("Correct predictions", f"{correct_count} / {len(results_df)}")
                st.metric("Intent accuracy", f"{accuracy:.2%}")

                st.subheader("Misclassified messages")
                errors_df = results_df[~results_df["correct"]]
                if errors_df.empty:
                    st.success("All messages were classified correctly.")
                else:
                    st.dataframe(errors_df, use_container_width=True)

                st.download_button(
                    "Download evaluation results",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="dessaint_app_evaluation_results.csv",
                    mime="text/csv",
                )

            except Exception as error:
                st.error("Evaluation failed.")
                st.exception(error)
