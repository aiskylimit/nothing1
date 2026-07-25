from math_verify import parse, verify
from utils import ANSWER_FORCE_STRING




def is_correct(trace: str, soln: str):
    """
    Evaluate if a generated trace produces the correct answer for a math problem.
    
    Uses math_verify to parse and compare solutions. Handles cases where the 
    answer forcing string splits the response.
    """

    try:
        if ANSWER_FORCE_STRING in trace:
            # Handle answer forcing: try multiple ways to extract the answer
            parts = trace.split(ANSWER_FORCE_STRING)
            alt_ans1 = ANSWER_FORCE_STRING.join(parts[:-1])
            alt_ans2 = parts[-1]
            res = any(verify(soln, parse(ans)) for ans in [trace, alt_ans1, alt_ans2])
        else:
            res = verify(soln, parse(trace))
    except:
        print(f"Error parsing trace: {trace} and comparing with solution: {soln}")
        res = False
    return res

def compute_math_metric(predictions, references):
    acc = 0
    for pred, ref in zip(predictions, references):
        if is_correct(pred, ref):
            acc += 1
    return {"Accuracy": acc / len(predictions)}