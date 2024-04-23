import streamlit as st
import streamlit_authenticator as stauth
import anthropic
import yaml
from yaml.loader import SafeLoader

with open('auth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
    print(config)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

def logout():
    authenticator.logout()


if __name__ == '__main__':

    authenticator.login()

    if st.session_state["authentication_status"]:
        if "name" in st.session_state:
            st.sidebar.write(f'Welcome *{st.session_state["name"]}*')
            
            uploaded_file = st.file_uploader("Upload an article", type=("txt", "md", "pdf"))

            if uploaded_file is not None:

                file_contents = uploaded_file.read()

                with open("files/"+ st.session_state["username"] + "_"+ uploaded_file.name, "wb") as f:
                    f.write(file_contents)

                st.success("File saved successfully!")
            
            if prompt := st.chat_input("Say something"):
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})


            if uploaded_file and prompt:
                article = uploaded_file.read().decode()
                prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n
                {article}\n\n\n\n{prompt}{anthropic.AI_PROMPT}"""

                client = anthropic.Client(api_key="")
                response = client.completions.create(
                    prompt=prompt,
                    stop_sequences=[anthropic.HUMAN_PROMPT],
                    model="claude-v1", #"claude-2" for Claude 2 model
                    max_tokens_to_sample=100,
                )
                st.write("### Answer")
                st.write(response.completion)

            
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])    

            # logout()
        else:
            st.sidebar.write("Please log in to view the content.")
            st.sidebar.title("Atlee Personalized AI")

        
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
        if st.session_state["authentication_status"]:
            try:
                if authenticator.reset_password(st.session_state["username"]):
                    st.success('Password modified successfully')
            except Exception as e:
                st.error(e)
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')
    