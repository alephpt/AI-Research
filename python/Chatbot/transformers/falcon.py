from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

def main():

    pipeline = transformers.pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        # use local code
        # local_files_only=True,
        trust_remote_code=True,
        device_map="auto"
        )

    userIn = input("Enter a prompt: ")

    while (userIn != "exit"):
        sequences = pipeline(
            userIn,
            max_length=500,
            do_sample = True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )

        print('Falcon: {}'.format(sequences[0]['generated_text']))
        print("-------------------------")
        userIn = input("Enter a prompt: ")


if __name__ == "__main__":
    main()


#
