import streamlit as st
from langchain_core.messages import HumanMessage
from rag_agent_trial import rag_agent  # import your backend agent

st.set_page_config(page_title="Stock Market RAG Assistant", page_icon="📊")
st.title("📊 Stock Market RAG Assistant")

# Keep conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input box
user_input = st.text_input("Ask a question about the Stock Market PDF:")
if st.button("Ask") and user_input:
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Run RAG agent
    result = rag_agent.invoke({"messages": st.session_state.messages})
    response = result['messages'][-1].content

    # Add AI response
    st.session_state.messages.append(result['messages'][-1])

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.markdown(f"👤 **You:** {msg.content}")
    else:
        st.markdown(f"🤖 **Assistant:** {msg.content}")
