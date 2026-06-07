import gradio as gr
import pickle
import numpy as np

# Load Models
with open("solar_flare_rf.pkl", "rb") as f:
    rf = pickle.load(f)

with open("solar_flare_mlp.pkl", "rb") as f:
    mlp = pickle.load(f)

with open("solar_flare_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("solar_flare_encoder.pkl", "rb") as f:
    le = pickle.load(f)


def predict_flare(
    active_region,
    linked_events,
    duration,
    rise_time,
    month,
    hour,
    model_choice
):

    X = np.array([[
        active_region,
        linked_events,
        duration,
        rise_time,
        month,
        hour
    ]])

    X_scaled = scaler.transform(X)

    if model_choice == "Random Forest":
        probabilities = rf.predict_proba(X_scaled)[0]
    else:
        probabilities = mlp.predict_proba(X_scaled)[0]

    predicted_index = np.argmax(probabilities)

    predicted_class = le.inverse_transform(
        [predicted_index]
    )[0]

    confidence = probabilities[predicted_index] * 100

    result = f"""
# ☀️ Solar Flare Prediction

### Predicted Flare Class
**{predicted_class}**

### Confidence
**{confidence:.2f}%**

### Class Probabilities
"""

    for cls, prob in zip(le.classes_, probabilities):
        result += f"\n- {cls}: {prob*100:.2f}%"

    return result


with gr.Blocks(title="NASA Solar Flare Predictor") as app:

    gr.Markdown("""
    # ☀️ NASA Solar Flare Predictor

    Predict Solar Flare Class using
    Machine Learning and Neural Networks.
    """)

    with gr.Row():

        with gr.Column():

            active_region = gr.Number(
                label="Active Region Number",
                value=12000
            )

            linked_events = gr.Dropdown(
                choices=[0, 1],
                value=0,
                label="Linked Events"
            )

            duration = gr.Number(
                label="Duration (Minutes)",
                value=10
            )

            rise_time = gr.Number(
                label="Rise Time (Minutes)",
                value=5
            )

            month = gr.Slider(
                minimum=1,
                maximum=12,
                value=6,
                step=1,
                label="Month"
            )

            hour = gr.Slider(
                minimum=0,
                maximum=23,
                value=12,
                step=1,
                label="Hour"
            )

            model_choice = gr.Radio(
                choices=[
                    "Random Forest",
                    "MLP Neural Network"
                ],
                value="Random Forest",
                label="Select Model"
            )

            predict_btn = gr.Button(
                "Predict Solar Flare",
                variant="primary"
            )

        with gr.Column():

            output = gr.Markdown()

    predict_btn.click(
        fn=predict_flare,
        inputs=[
            active_region,
            linked_events,
            duration,
            rise_time,
            month,
            hour,
            model_choice
        ],
        outputs=output
    )

app.launch()