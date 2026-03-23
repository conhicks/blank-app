import streamlit as st
from openai import OpenAI
from pydantic import BaseModel
import json

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Pydantic model for structured summary
class StructuredSummary(BaseModel):
    main_answer: str
    key_facts: list[str]
    source_hint: str

# Initialize session state
if "last_response_id" not in st.session_state:
    st.session_state.last_response_id = None

# Sidebar checkboxes
structured_summary = st.sidebar.checkbox("Return structured summary")
enable_streaming = st.sidebar.checkbox("Enable streaming")

st.title("Lab6: OpenAI Responses API App")

# Main form for initial question
with st.form("initial_form"):
    question = st.text_input("Enter your question:")
    submitted = st.form_submit_button("Submit")

    if submitted and question:
        # Reset last_response_id for new question
        st.session_state.last_response_id = None

        if structured_summary:
            # Use parse for structured summary
            response = client.responses.parse(
                model="gpt-4o",
                instructions="You are a helpful research assistant. Always cite your sources.",
                input=question,
                text_format={"type": "json_schema", "json_schema": StructuredSummary.model_json_schema()}
            )
            # Parse the response
            parsed = response.output_text
            summary = StructuredSummary.model_validate_json(parsed)
            st.write(summary.main_answer)
            st.write("Key facts:")
            for fact in summary.key_facts:
                st.write(f"- {fact}")
            st.caption(summary.source_hint)
        else:
            # Use create
            if enable_streaming:
                # Streaming
                placeholder = st.empty()
                full_text = ""
                with client.responses.create(
                    model="gpt-4o",
                    instructions="You are a helpful research assistant. Always cite your sources.",
                    input=question,
                    tools=[{"type": "web_search_preview"}],
                    stream=True
                ) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            full_text += event.delta
                            placeholder.write(full_text)
                        elif event.type == "response.completed":
                            st.session_state.last_response_id = event.response.id
            else:
                # Non-streaming
                response = client.responses.create(
                    model="gpt-4o",
                    instructions="You are a helpful research assistant. Always cite your sources.",
                    input=question,
                    tools=[{"type": "web_search_preview"}]
                )
                st.write(response.output_text)
                st.session_state.last_response_id = response.id

        st.caption("Web search is enabled.")

# Follow-up form, only if there's a last_response_id
if st.session_state.last_response_id:
    with st.form("followup_form"):
        followup_question = st.text_input("Enter a follow-up question:")
        followup_submitted = st.form_submit_button("Submit Follow-up")

        if followup_submitted and followup_question:
            if structured_summary:
                # Use parse for follow-up
                response = client.responses.parse(
                    model="gpt-4o",
                    instructions="You are a helpful research assistant. Always cite your sources.",
                    input=followup_question,
                    previous_response_id=st.session_state.last_response_id,
                    text_format={"type": "json_schema", "json_schema": StructuredSummary.model_json_schema()}
                )
                parsed = response.output_text
                summary = StructuredSummary.model_validate_json(parsed)
                st.write(summary.main_answer)
                st.write("Key facts:")
                for fact in summary.key_facts:
                    st.write(f"- {fact}")
                st.caption(summary.source_hint)
            else:
                # Use create for follow-up
                if enable_streaming:
                    # Streaming follow-up
                    placeholder = st.empty()
                    full_text = ""
                    with client.responses.create(
                        model="gpt-4o",
                        instructions="You are a helpful research assistant. Always cite your sources.",
                        input=followup_question,
                        previous_response_id=st.session_state.last_response_id,
                        tools=[{"type": "web_search_preview"}],
                        stream=True
                    ) as stream:
                        for event in stream:
                            if event.type == "response.output_text.delta":
                                full_text += event.delta
                                placeholder.write(full_text)
                            elif event.type == "response.completed":
                                st.session_state.last_response_id = event.response.id
                else:
                    # Non-streaming follow-up
                    response = client.responses.create(
                        model="gpt-4o",
                        instructions="You are a helpful research assistant. Always cite your sources.",
                        input=followup_question,
                        previous_response_id=st.session_state.last_response_id,
                        tools=[{"type": "web_search_preview"}]
                    )
                    st.write(response.output_text)
                    st.session_state.last_response_id = response.id

            st.caption("Web search is enabled.")