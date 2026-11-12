import streamlit as st

st.set_page_config(page_title="AirQuality-SLM", page_icon="🌿")
st.title("AirQuality-SLM")
st.caption("Historical measurement interpretation with a fine-tuned FLAN-T5-small checkpoint")
model_path = st.sidebar.text_input("Model path", "artifacts/flan-t5-airquality/best_model")
task = st.selectbox("Task", ["Classify", "Explain"])
co = st.number_input("CO (mg/m³)", value=2.0)
no2 = st.number_input("NO₂ (µg/m³)", value=100.0)
benzene = st.number_input("Benzene (µg/m³)", value=10.0)

if st.button("Generate interpretation"):
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        prefix = ("Classify these historical air-quality measurements as low, typical, elevated or unavailable "
                  "relative to training-distribution quartiles. " if task == "Classify" else
                  "Explain these historical air-quality measurements in one factual sentence. Use relative historical "
                  "quartiles only; do not infer official AQI, health safety or a pollution source. ")
        prompt = prefix + f"CO: {co:.2f} mg/m3; NO2: {no2:.2f} ug/m3; benzene: {benzene:.2f} ug/m3."
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=64)
        st.success(tokenizer.decode(output[0], skip_special_tokens=True))
        st.code(prompt, language=None)
    except Exception as exc:
        st.error(f"Inference unavailable: {exc}")
        st.info("Install requirements and train the checkpoint first. The app does not substitute rule-based output for a missing model.")
