from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

def main():
    generation_config = transformers.GenerationConfig.from_pretrained(
        model_name,
        top_k=1,
        return_unused_kwargs=True
    )

    pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, config=generation_config)

    userIn = input("Enter a prompt: ")

    while (userIn != "exit"):
        sequences = pipeline('Your name is Glaza. You have been trained on the Falcon-7B-Instruct model, and are the central bot to help assist in the management of Order 332.' +
                             'Order 332 is a discord server that embraces cultures from all parts of the world, and values education and open-mindedness. The Order is based around' +
                             ' the pillars of Community, Knowledge and Friendship. We aspire to maintain brutal honesty, openness, and full disclosure.. even it it means ' +
                             ' being brutally honest, and saying things that people may not here. Please adhere to these values. What say you of the following query: ' + userIn,
                             max_length=250, num_return_sequences=1)
        print('Falcon: {}'.format(sequences[0]['generated_text']))
        print("-------------------------")
        userIn = input("Enter a prompt: ")


if __name__ == "__main__":
    main()
