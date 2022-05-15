import streamlit as st
import tensorflow
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained('t5-small')
import numpy as np

#@st.cache(allow_output_mutation=True)
@st.experimental_singleton
def get_model():
    model=TFT5ForConditionalGeneration.from_pretrained('t5-small')
    return model


def find_answer(context):
    m=get_model()
    text='Summarize: ' + context
    input_ids = tokenizer.encode(text, return_tensors='tf')
    beam_output = m.generate(input_ids,max_length = 100,num_beams=5,temperature=0.7)
    output=tokenizer.decode(beam_output[0],skip_special_tokens=True)
    return output

st.title("Text Summarisation")
st.subheader("Example Text:")
st.markdown("The empire was established by Chandragupta Maurya assisted by Chanakya (Kautilya) in Magadha (in modern Bihar) when he overthrew the Nanda dynasty. Chandragupta rapidly expanded his power westwards across central and western India, and by 317 BCE the empire had fully occupied Northwestern India.")
st.subheader("Summary:")
st.markdown("the empire was established by Chandragupta Maurya in Magadha. by 317 BCE the empire had fully occupied Northwestern India.")

st.markdown("\n")
form = st.form(key="form")
context = form.text_area("Enter the text here")

predict_button = form.form_submit_button(label='Submit')


if predict_button:
    st.write("Response:")
    with st.spinner('Finding Answer'):
        answer = find_answer(context)
        st.write("Summary:",answer)


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by <a style='display: block; text-align: center;' href="https://www.linkedin.com/in/duddukunta-sasidhar-reddy-3396781b5/" target="_blank">Duddukunts Sashidhar Reddy</a></p>
<p>Email ID : <a style='display: block; text-align: center;' target="_blank">dsasidharreddy867@gmail.com</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)