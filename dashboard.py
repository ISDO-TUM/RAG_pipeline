import streamlit as st
from dashboard_utils import *
import plotly.express as px
import datetime
import pandas as pd
from st_keyup import st_keyup


RETRIEVAL_METHODS = ["Nearest Neighbor", "Contextual Compression", "SVM", " Multi-Query", "Ensemble"]
EMBEDDING_MODEL = ["text-embedding-3-small", "text-embedding-3-large", "ada v2", "HuggingFaceEmbeddings"]
TEXTSPLITTER = ["RecursiveCharacterTextSplitter", "SemanticTextSplitter"]
TEXTSPLITTER_SEMANTIC_BREAKPOINT_TYPE = ["percentile", "standard_deviation", "interquartile"]
MODEL = ["gpt-3.5-turbo"]
PROMPT = ["default"]


def main():
    st.set_page_config(layout="wide")

    # Initialize session states
    if 'metrics' not in st.session_state:
        st.session_state['metrics'] = None
    if 'results' not in st.session_state:
        st.session_state['results'] = None      

    # Create columns for the header
    col1, col2 = st.columns([4, 2])
    with col1:
        st.title("Dashboard")
        display_previous_runs(col1)
    with col2:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        
    # Process the uploaded file
    if uploaded_file is not None:
        with col2:
            st.subheader("Configuration")
            if uploaded_file is not None:
                config_params = create_tabs()
        with col2:
            save_results = st.checkbox("Save Results")
            test_name = st.text_input("Test name")
            calculate_button = st.button("Calculate Metrics")

            if calculate_button and test_name:
                # reset session state to display new results
                st.session_state["metrics"] = None
                st.session_state["results"] = None
                
                timestamp = datetime.datetime.now()
                config = update_config(config_params)
                with st.spinner("Wait for it..."):
                    results = display_metrics(
                        col1, uploaded_file=uploaded_file
                    )
                display_heatmap(col1, results)
                
                # Save results to MongoDB if checkbox is checked
                if save_results:
                    insert_document(test_name, timestamp, config, results)
                    
    # Display the results if they exist in the database                
    if st.session_state["results"]:
        display_metrics(col1)
        display_heatmap(col1)
        
        st.subheader("Corresponding Configuration")
        st.write(st.session_state["config"])
                    

def login() -> bool:
    st.title("Login in to your account to access the dashboard")
    email = st.text_input("Enter your email")
    password = st.text_input("Enter your password", type="password")
    login_button = st.button("Login")

    if login_button:
        with st.spinner("Logging in..."):
            try:
                user = authenticate_user(email=email, password=password)
                st.success(f"Login successful! Welcome {user['email']}")
                return True
            except requests.exceptions.HTTPError as e:
                st.error(
                    f"Login failed: {e.response.json()['error']['message']}")
                return False


def display_metrics(col1, uploaded_file=None):
    if not st.session_state["metrics"]:
        metrics = call_pipeline(uploaded_file)
    else:
        metrics = st.session_state["metrics"]

    with col1:
        st.subheader("Metrics")
        context_recall = f"{metrics['context_recall']:.2%}"
        context_precision = f"{metrics['context_precision']:.2%}"
        faithfulness = f"{metrics['faithfulness']:.2%}"
        answer_relevancy = f"{metrics['answer_relevancy']:.2%}"

        # Create columns in Streamlit
        m1, m2, m3, m4 = st.columns(4)

        # Display metrics in Streamlit columns
        m1.metric(
            "Context Recall",
            context_recall,
            "",
        )
        m2.metric(
            "Context Precision",
            context_precision,
            "",
        )
        m3.metric(
            "Faithfulness",
            faithfulness,
            "",
        )
        m4.metric(
            "Answer Relevancy",
            answer_relevancy,
            "",
        )
    return metrics


def display_heatmap(col1, results=None):
    with col1:
        if not st.session_state["results"]:
            results = results.to_pandas()
        else:
            results = pd.DataFrame(st.session_state["results"])
        
        results = results.map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        results = results.map(lambda x: '[' + '],\r\n\r\n ['.join(x) + ']' if isinstance(x, list) else x)
        
        matrix = results[
            ["context_recall", "context_precision",
                "faithfulness", "answer_relevancy"]
        ].values.tolist()

        fig = px.imshow(
            matrix,
            text_auto=True,
            aspect="auto",
            x=[
                "context_recall",
                "context_precision",
                "faithfulness",
                "answer_relevancy",
            ],
            y=list(map(lambda x: 4*' ' + f'Question {str(x)}', list(
                range(len(results["question"].values)))))
        )

        st.plotly_chart(fig, theme="streamlit")
        
        st.subheader("Retrieved Dataframe")
        st.write(results)


def create_tabs():
    tab1, tab2, tab3 = st.tabs(["Indexing", "Retrieval", "Generation"])

    # Content for the "Indexing" tab
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            embedding = st.selectbox("Embedding model", EMBEDDING_MODEL)
        with col2:
            textsplitter = st.selectbox("TextSplitter", TEXTSPLITTER)
            
        if textsplitter == "SemanticTextSplitter":
            with col3:
                breakpoint_type = st.selectbox(
                    "Breakpoint Type", TEXTSPLITTER_SEMANTIC_BREAKPOINT_TYPE
                )
            st.write(f"Indexing: {embedding}, {textsplitter}, {breakpoint_type}")
        else:
            with col3:
                chunksize = st.number_input(
                    "Enter chunk size", min_value=10, max_value=2000, value=200, step=50
                )
            st.write(f"Indexing: {embedding}, {textsplitter}, {chunksize}")

    # Content for the "Retrieval" tab
    with tab2:
        col4, col5, col6 = st.columns(3)
        with col4:
            retrieval_method = st.selectbox(
                "Retrieval Method", RETRIEVAL_METHODS)
        with col5:
            k = st.number_input(
                "Enter k number of retrieved chunks",
                min_value=1,
                max_value=15,
                value=5,
                step=1,
            )
        with col6:
            reranker_id = st.number_input(
                "Enter reranker id",
                min_value=0,
                max_value=4,
                value=0,
                step=1,
            )
        st.write(f"Retrieval: {retrieval_method}, {k}, {reranker_id}")

    # Content for the "Generation" tab
    with tab3:
        col7, col8, col9 = st.columns(3)
        with col7:
            model = st.selectbox("Model", MODEL)
        with col8:
            prompt = st.selectbox("Prompt", PROMPT)
        with col9:
            temperature = st.number_input(
                "Enter temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1
            )
        st.write(f"Generation: {model}, {prompt}, {temperature}")

    return [
        embedding,
        textsplitter,
        chunksize if textsplitter != "SemanticTextSplitter" else breakpoint_type,
        retrieval_method,
        k,
        reranker_id,
        model,
        prompt,
        temperature,
    ]


def display_previous_runs(col1):
    num_results = 5
    
    with col1:
        with st.expander("See Previous Runs"):
            search_query = st_keyup("Search for test name:")
            
            # Header
            colms = st.columns((1, 1, 1, 1))
            fields = ["id", 'test_name', 'timestamp', 'open']
            for col, field_name in zip(colms, fields):
                col.write(field_name)
                
            data=get_data(search_query)[:num_results]
            
            if data is None:
                st.write("No data found")
            
            # Display data
            for id, item in enumerate(data):
                # Extract values
                test_name = item['test_name']
                timestamp = item['timestamp']
                config = item['config']
                metrics = item['results']['average_score_per_metric']
                results = item['results']['results_per_question']
                
                col1, col2, col3, col4 = st.columns((1, 1, 1, 1))
                col1.write(id) 
                col2.write(test_name)  
                col3.write(timestamp) 
                if col4.button("Open", key=id):
                    # set the session state
                    st.session_state["config"] = config
                    st.session_state["metrics"] = metrics
                    st.session_state["results"] = results

if __name__ == "__main__":
    main()
