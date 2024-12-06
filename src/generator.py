# src/generator.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME_GENERATOR = "facebook/bart-large-cnn"

GENERATOR_PARAMS = {
    "max_length": 200,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
    "num_beams": 4
}


class Generator:
    def __init__(
            self,
            model_name : str = MODEL_NAME_GENERATOR
        ):
        """
        Initialize the Generator by loading the language model and tokenizer.
        """
        assert isinstance(model_name, str), "Model name must be a string"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def generate_answer(
            self,
            question : str,
            context : str,
            max_length : int = GENERATOR_PARAMS["max_length"]
        ):
        """
        Generate an answer based on the question and context.
        """
        assert isinstance(question, str), "Question must be a string"
        assert isinstance(context, str), "Context must be a string"
        assert isinstance(max_length, int), "max_length must be an integer"

        # Prepare the input text
        input_text = f"Question: {question}\nContext: {context}\nAnswer:"
        
        # Tokenize and encode the input text
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=1024
        )
        
        # Generate the output
        outputs = self.model.generate(
            inputs, 
            max_length=GENERATOR_PARAMS["max_length"],
            early_stopping=GENERATOR_PARAMS["early_stopping"],
            no_repeat_ngram_size=GENERATOR_PARAMS["no_repeat_ngram_size"],
            num_beams=GENERATOR_PARAMS["num_beams"]
        )
        
        # Decode the generated answer
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()
