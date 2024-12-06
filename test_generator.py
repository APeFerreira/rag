# src/test_generator.py

import os
import sys
from src.generator import Generator
import torch

def load_generator():
    """
    Initialize and return the Generator instance.
    """
    try:
        generator = Generator()
        return generator
    except Exception as e:
        print(f"Error loading Generator: {e}")
        sys.exit(1)

def generate_and_display(generator, question, context, max_length=200):
    """
    Generate an answer using the Generator and display the results.
    """
    try:
        answer = generator.generate_answer(question, context, max_length=max_length)
        print(f"Question: {question}\n")
        print(f"Context: {context}\n")
        print(f"Generated Answer: {answer}\n")
        print("-" * 80 + "\n")
    except Exception as e:
        print(f"Error generating answer: {e}")

def main():
    """
    Main function to execute the generator tests.
    """
    print("torch available: ", torch.cuda.is_available())
    # Initialize Generator
    print("Loading Generator...")
    generator = load_generator()
    print("Generator loaded successfully.\n")

    # Define sample questions and contexts based on provided case texts
    sample_tests = [
        {
            "question": "What criteria are used to assess apparent bias in judicial decisions?",
            "context": (
                "An alternative ground for the conclusion that I should not disqualify myself from hearing the present application "
                "on the ground of apparent bias is this. The test for apparent bias is well-known and it is 'whether a fair-minded lay "
                "observer might reasonably apprehend that the judge might not bring an impartial and unprejudiced mind to the resolution "
                "of the question the judge is required to decide': Johnson v Johnson (2000) 201 CLR 488 at 492 [11] per Gleeson CJ, "
                "Gaudron, McHugh, Gummow and Hayne JJ. In this case, if there is apparent bias, it takes the form of prejudgment..."
            )
        },
        {
            "question": "Under what circumstances can a court issue cost orders without proceeding to trial?",
            "context": (
                "Mr Spencer submitted that he had sought on a number of occasions to suggest a reasonable basis upon which this proceeding could "
                "be settled, and that the ACCC had acted unreasonably in refusing to engage with him. He referred to Australian Securities Commission "
                "v Aust-Home Investments Limited (1993) 44 FCR 194 at 201 where Hill J summarised the principles that would ordinarily apply "
                "with regard to costs when a proceeding has not gone to trial..."
            )
        },
        {
            "question": "When are indemnity costs awarded instead of party and party costs in court proceedings?",
            "context": (
                "Ordinarily that discretion will be exercised so that costs follow the event and are awarded on a party and party basis. A departure "
                "from normal practice to award indemnity costs requires some special or unusual feature in the case: Alpine Hardwood (Aust) Pty Ltd "
                "v Hardys Pty Ltd (No 2) [2002] FCA 224 ; (2002) 190 ALR 121 at [11] (Weinberg J) citing Colgate Palmolive Co v Cussons Pty Ltd "
                "(1993) 46 FCR 225 at 233 (Sheppard J)."
            )
        }
    ]

    # Execute tests
    for test in sample_tests:
        generate_and_display(
            generator=generator,
            question=test["question"],
            context=test["context"],
            max_length=200  # Adjust as needed
        )

if __name__ == "__main__":
    main()
