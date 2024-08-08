import configparser
from flask import Flask, render_template, request
from rag.functions.vector_indexing import get_vectordb
from rag.pipeline import Pipeline
from rag.models.dataloader import DataLoader
import markdown

# Load confing
config = configparser.ConfigParser(interpolation=None)
config.read('config.ini')

# Clear the logs
with open(config["logging"]["filename"], 'w') as file:
    pass 

# Create a flask app instance
app = Flask(__name__)

# Initial conversation
conversation = []

data_loader = DataLoader(config["ingestion"])
# Build database and initialize pipeline
vectordb = get_vectordb(config, data_loader)
pipeline = Pipeline(vectordb=vectordb, config=config)



@app.route('/')
def home():
    """
    The frontend ui with a persistent conversation

    Returns:
         frontend HTML
    """
    return render_template('index.html', conversation=conversation)


@app.route('/logs')
def logs():
    with open(config["logging"]["filename"], 'r') as log_file:
        log_entries = []
        for line in log_file:
            parts = line.split(' - ')
            if len(parts) == 4:
                log_entries.append({
                    'asctime': parts[0],
                    'name': parts[1],
                    'levelname': parts[1],
                    'message': parts[3].strip()
                })

        return render_template('logs.html', logs= log_entries)

@app.route('/ask', methods=['POST'])
def ask():
    """
    POST Endpoint to ask a question. The response is returned as json and the conversation for the UI is updated.

    Returns:
        {"question": question, "answer": answer, "documents": documents}
    """
    if request.method == 'POST':
        question = request.form['question']

        flattened_conversation = [msg for pair in conversation for msg in list(pair[:2])]
        answer, documents = pipeline.invoke(question, flattened_conversation)
        answer = markdown.markdown(answer)
        
        conversation.append((question, answer, documents))

        return {}, 200
    else:
        return {"response": "method not allowed"}, 405


if __name__ == '__main__':
    """
    Entry point for the web server
    """
    host = config["web"]["host"]
    port = int(config["web"]["port"])
    app.run(debug=True, host=host, port=port)
