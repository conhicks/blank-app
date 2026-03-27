import streamlit as st
from openai import OpenAI
from pydantic import BaseModel

# --- Page Config ---
st.set_page_config(page_title="Lab 6 - Responses API Agent", layout="centered")
st.title("Research Assistant")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.sidebar.header("Agent Settings")
use_structured = st.sidebar.checkbox("Return structured summary")
use_streaming = st.sidebar.checkbox("Enable streaming")

class ResearchSummary(BaseModel):
    main_answer: str
    key_facts: list[str]
    source_hint: str

if "last_response_id" not in st.session_state:
    st.session_state.last_response_id = None
if "first_response_text" not in st.session_state:
    st.session_state.first_response_text = None

def get_tools():
    return [{"type": "web_search_preview"}]

def call_responses_api(user_input, previous_response_id=None, structured=False, streaming=False):
    instructions = "You are a helpful research assistant. Always cite your sources when using web search."

    kwargs = dict(
        model="gpt-4o",
        instructions=instructions,
        input=user_input,
        tools=get_tools(),
    )
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    if structured:
        response = client.responses.parse(
            **{k: v for k, v in kwargs.items() if k != "tools"}, text_format=ResearchSummary,
        )
        return response, None

    if streaming:
        stream = client.responses.create(**kwargs, stream=True)
        return None, stream

    response = client.responses.create(**kwargs)
    return response, None


st.subheader("Ask a Question")

with st.form("initial_form"):
    user_question = st.text_input("Your question:", placeholder="e.g. What are the latest AI news headlines?")
    submitted = st.form_submit_button("Ask")

if submitted and user_question:
    st.session_state.last_response_id = None
    st.session_state.first_response_text = None

    with st.spinner("Thinking..."):
        if use_structured:
            response, _ = call_responses_api(user_question, structured=True)
            parsed: ResearchSummary = response.output_parsed
            st.session_state.last_response_id = response.id
            st.session_state.first_response_text = parsed.main_answer

            st.markdown("### Answer")
            st.write(parsed.main_answer)
            st.markdown("**Key Facts:**")
            for fact in parsed.key_facts:
                st.markdown(f"- {fact}")
            st.caption(f"Source hint: {parsed.source_hint}")

        elif use_streaming:
            _, stream = call_responses_api(user_question, streaming=True)
            st.markdown("### Answer")
            response_text = ""
            placeholder = st.empty()
            response_id = None
            for event in stream:
                if event.type == "response.output_text.delta":
                    response_text += event.delta
                    placeholder.markdown(response_text)
                if event.type == "response.completed":
                    response_id = event.response.id
            if response_id:
                st.session_state.last_response_id = response_id
                st.session_state.first_response_text = response_text

        else:
            response, _ = call_responses_api(user_question)
            st.session_state.last_response_id = response.id
            st.session_state.first_response_text = response.output_text

            st.markdown("### Answer")
            st.write(response.output_text)


if st.session_state.last_response_id:
    st.divider()
    st.subheader("Ask a Follow-Up Question")
    st.info("The agent remembers your previous question so theres no need to repeat it.")

    with st.form("followup_form"):
        followup = st.text_input(
            "Follow-up:",
            placeholder="e.g. Can you give me more detail on the second point?"
        )
        followup_submitted = st.form_submit_button("Ask Follow-Up")

    if followup_submitted and followup:
        with st.spinner("Thinking..."):
            prev_id = st.session_state.last_response_id

            if use_structured:
                response, _ = call_responses_api(followup, previous_response_id=prev_id, structured=True)
                parsed: ResearchSummary = response.output_parsed
                st.session_state.last_response_id = response.id

                st.markdown("### Follow-Up Answer")
                st.write(parsed.main_answer)
                st.markdown("**Key Facts:**")
                for fact in parsed.key_facts:
                    st.markdown(f"- {fact}")
                st.caption(f"Source hint: {parsed.source_hint}")

            elif use_streaming:
                _, stream = call_responses_api(followup, previous_response_id=prev_id, streaming=True)
                st.markdown("### Follow-Up Answer")
                response_text = ""
                placeholder = st.empty()
                response_id = None
                for event in stream:
                    if event.type == "response.output_text.delta":
                        response_text += event.delta
                        placeholder.markdown(response_text)
                    if event.type == "response.completed":
                        response_id = event.response.id
                if response_id:
                    st.session_state.last_response_id = response_id

            else:
                response, _ = call_responses_api(followup, previous_response_id=prev_id)
                st.session_state.last_response_id = response.id
                st.markdown("### Follow-Up Answer")
                st.write(response.output_text)