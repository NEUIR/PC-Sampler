
def humaneval_prompt(func): # prompt for humaneval
     prompt = f'''Role: You are a professional Python coding assistant
Task: Complete the follow function implementation strictly and clearly without any additional comments or explanations.
{func}'''
     return prompt

def mbpp_prompt(func, code): # prompt for mbpp
    prompt = f'''Role: You are a professional Python coding assistant
Task: Complete the follow function implementation strictly and clearly without any additional comments or explanations.
{func}
the function name and the parameters is {code.split('def')[-1].split(':')[0]}
'''
    return prompt

def math_500_prompt(question): # prompt for math-500
    prompt = r'''Q: Let \[f(x) = \left\{
\begin{array}{cl} ax+3, &\text{ if }x>2, \\
x-5 &\text{ if } -2 \le x \le 2, \\
2x-b &\text{ if } x <-2.
\end{array}
\right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).

  A: <think>
For the piecewise function to be continuous, the cases must "meet" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \Rightarrow a=-3$. Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\boxed{0}$.
</think>
<answer>
0
</answer>

Q: A rectangular band formation is a formation with $m$ band members in each of $r$ rows, where $m$ and $r$ are integers. A particular band has less than 100 band members. The director arranges them in a rectangular formation and finds that he has two members left over. If he increases the number of members in each row by 1 and reduces the number of rows by 2, there are exactly enough places in the new formation for each band member. What is the largest number of members the band could have?

  A: <think>
Let $x$ be the number of band members in each row for the original formation, when two are left over. Then we can write two equations from the given information: $$rx+2=m$$ $$(r-2)(x+1)=m$$ Setting these equal, we find: $$rx+2=(r-2)(x+1)=rx-2x+r-2$$ $$2=-2x+r-2$$ $$4=r-2x$$ We know that the band has less than 100 members. Based on the first equation, we must have $rx$ less than 98. We can guess and check some values of $r$ and $x$ in the last equation. If $r=18$, then $x=7$, and $rx=126$ which is too big. If $r=16$, then $x=6$, and $rx=96$, which is less than 98. Checking back in the second formation, we see that $(16-2)(6+1)=14\cdot 7=98$ as it should. This is the best we can do, so the largest number of members the band could have is $\boxed{98}$.
</think>
<answer>
98
</answer>

Q: What is the degree of the polynomial $(4 +5x^3 +100 +2\pi x^4 + \sqrt{10}x^4 +9)$?

  A: <think>
This polynomial is not written in standard form. However, we don't need to write it in standard form, nor do we need to pay attention to the coefficients. We just look for the exponents on $x$. We have an $x^4$ term and no other term of higher degree, so $\boxed{4}$ is the degree of the polynomial.
</think>
<answer>
4
</answer>

Q: Evaluate $\left\lceil3\left(6-\frac12\right)\right\rceil$.

  A: <think>
Firstly, $3\left(6-\frac12\right)=18-1-\frac12=17-\frac12$. Because $0\le\frac12<1$, we have $\left\lceil17-\frac12\right\rceil=\boxed{17}$.
</think>
<answer>
17
</answer>

Q: {{question}}

  A:
'''
    return prompt.replace("{{question}}", question)

SUDOKU_SYSTEM_PROMPT = """
Solve this 4x4 Sudoku puzzle represented as a 16-digit string (read left-to-right, top-to-bottom) where '0'=empty cell.

Requirements:
1. Replace ALL '0's with digits 1-4
2. Follow STRICT Sudoku rules:
   - Rows: Each must contain 1-4 exactly once
   - Columns: Each must contain 1-4 exactly once
   - 2x2 Boxes: Each must contain 1-4 exactly once
3. Format answer as:
<answer>
[16-digit solution]
</answer>
"""

SUDOKU_PROMPT = """Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Here are some examples:


Puzzle: 
4100
0001
1300
2000
<answer> 
4132
3241
1324
2413
</answer>

Puzzle: 
0004
0321
0203
3002
<answer>
2134
4321
1243
3412
</answer>

Puzzle: 
4123
0000
0402
2300
<answer>
4123
3214
1432
2341
</answer>

Puzzle:
1432
0041
3000
4000
<answer>
1432
2341
3214
4123
</answer>

Puzzle:
0020
0341
0210
1002
<answer>
4123
2341
3214
1432
</answer>

Puzzle: {puzzle_str}
<answer>

"""

def sudoku_prompt(puzzle_str: str): # prompt for sudoku
    
    puzzle_str = '\n'.join(puzzle_str[i:i+4] for i in range(0, len(puzzle_str), 4))
    question = SUDOKU_PROMPT.format(puzzle_str=puzzle_str)
    
    
    return SUDOKU_SYSTEM_PROMPT + '\n\n' + question

def countdown_prompt(question): # prompt for countdown
    prompt = f'''For the given numbers, find a sequence of arithmetic operations that results in the target number.
Show your reasoning and conclude with "The answer is: [formula]".

Examples:
question: 15,44,79,50
Solution: Let's try to combine 15 and 44. 44 - 15 = 29. Now we have 29 and the remaining number 79. We need to reach the target 50. Let's try 79 - 29 = 50. This works. The answer is: 44-15=29,79-29=50

question: 1,2,12,25
Solution: We have 1, 2, 12 and the target is 25. Let's try multiplying 2 and 12. 2 * 12 = 24. Now we have 24 and the remaining number 1. We need to reach 25. 24 + 1 = 25. This is correct. The answer is: 2*12=24,1+24=25

question: 3,85,5,30
Solution: The numbers are 3, 85, 5 and the target is 30. Let's try adding 85 and 5. 85 + 5 = 90. Now we have 90 and the remaining number 3. We need to reach 30. 90 / 3 = 30. That's the target. The answer is: 85+5=90,90/3=30

New Question: {question}
Solution: '''
    return prompt