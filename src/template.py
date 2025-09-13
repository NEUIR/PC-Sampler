
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
  
def gsm8k_prompt(question):
    prompt = f'''Please solve the new question step by step just like the following examples. For this question:
1. Break down the problem into logical steps
2. Show all intermediate calculations
3. Conclude with "So the answer is..." format

Examples:
question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
target: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
target: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

New Question: {question}
Solution: '''
    return prompt

def gpqa_prompt(question, choice1, choice2, choice3, choice4):
    shot1 = '''Here are some example questions from experts. Answer the final question yourself, following the format of the previous questions exactly.
Question: A large gene has dozens of exons, of which the central ones code for folded triple helical repeats that connect the cytoskeleton with sarcolemma and extracellular space. Each exon usually codes for one folded triple alpha helix. The most common mutations of the gene are central exon deletions that create out-of-frame peptides and progressive degenerative organ waste. A solution is to deliver a Morpholino that recognizes the 5' end of the out-of-frame exon in pre-mRNA. The molecule prevents binding of the spliceosome and creates exon skipping and in-frame joining. Several missing exons are well tolerated by an organism. Which structure below is not involved in the proposed therapy?
Choices:
(A) lariat
(B) polyA tail
(C) antisense
(D) R-loops
Thought Process: The text describes the dystrophin gene and the FDA-approved oligonucleotide therapy that causes exon skipping by creating a functional, albeit shorter, dystrophin protein. Morpholino is bound to the pre-mRNA in an antisense orientation. Every splicing mechanism creates the lariat molecule that is circular with a 3' tail and soon degraded. The spliced RNA is polyadenylated at the 3' end. R-loops are triple helix of DNA and the pre-mRNA and a consequence of the RNA transcription, not splicing and RNA maturation. The answer is (D)'''
    shot2 = '''Question: Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?
Choices:
(A) 10^-11 eV
(B) 10^-8 eV
(C) 10^-9 eV
(D) 10^-4 eV
Thought Process: According to the uncertainty principle, Delta E* Delta t=hbar/2. Delta t is the lifetime and Delta E is the width of the energy level. With Delta t=10^-9 s==> Delta E1= 3.3 10^-7 ev. And Delta t=10^-11 s gives Delta E2=3.310^-8 eV. Therefore, the energy difference between the two states must be significantly greater than 10^-7 ev. So the answer is 10^-4 ev. The answer is (D)'''
    shot3 = '''Question: trans - cinnamaldehyde was treated with methylmagnesium bromide, forming product 1.
1 was treated with pyridinium chlorochromate, forming product 2.
3 was treated with (dimethyl(oxo)-l6 - sulfaneylidene)methane in DMSO at elevated temperature, forming product 3.
how many carbon atoms are there in product 3?
Choices:
(A) 10
(B) 11
(C) 12
(D) 14
Thought Process: trans - cinnamaldehyde was treated with methylmagnesium bromide, forming (E)-4 - phenylbut - 3 - en - 2 - ol. (E)-4 - phenylbut - 3 - en - 2 - ol was treated with pyridinium chlorochromate, forming (E)-4 - phenylbut - 3 - en - 2 - one. (E)-4 - phenylbut - 3 - en - 2 - one was treated with (dimethyl(oxo)-l6 - sulfaneylidene)methane in DMSO at elevated temperature, forming 1-(2 - phenylcyclopropyl)ethan - 1 - one. Chemical Formula: C11H12O. The answer is (B)'''
    shot4 = '''Question: how many of the following compounds exhibit optical activity?
1 - methyl - 4-(prop - 1 - en - 2 - yl)cyclohex - 1 - ene
2,3,3,3 - tetrafluoroprop - 1 - ene
di(cyclohex - 2 - en - 1 - ylidene)methane
5-(5 - methylhexan - 2 - ylidene)cyclopenta - 1,3 - diene
3-(2 - methylbut - 1 - en - 1 - ylidene)cyclohex - 1 - ene
[1,1'-biphenyl] - 3,3'-diol
8,8 - dichlorobicyclo[4.2.0]octan - 7 - one
cyclopent - 2 - en - 1 - one
Choices:
(A) 3
(B) 4
(C) 5
(D) 6
Thought Process: the compounds 1 - methyl - 4-(prop - 1 - en - 2 - yl)cyclohex - 1 - ene, 3-(2 - methylbut - 1 - en - 1 - ylidene)cyclohex - 1 - ene, di(cyclohex - 2 - en - 1 - ylidene)methane, 8,8 - dichlorobicyclo[4.2.0]octan - 7 - one are chiral molecules, and thus will be optically active. all the others have a mirror plane of symmetry, and will be achiral. The answer is (B)'''
    shot5 = '''Question: A coating is applied to a substrate resulting in a perfectly smooth surface. The measured contact angles of this smooth coating are 132° and 102° for water and hexadecane respectively. The coating formulation is then modified and when now applied to the same type of substrate, a rough surface is produced. When a droplet of water or oil sits on the rough surface, the wettability of the surface can now be described by the Cassie - Baxter state. The water contact angle on the rough surface is now 148°. What would be the best estimate of the contact angle of a droplet of octane on the rough surface? 
Choices:
(A) 129°
(B) 134°
(C) 139°
(D) 124°
Thought Process: In the Cassie - Baxter state, droplets are in contact with a non - uniform surface: some of the droplet is in contact with the coating and some with air. The contact angle (θCB) of a droplet in the Cassie - Baxter state is given by: cosθCB = f1.cosθ1 + f2.cosθ2. Where f1 and f2 are the area fractions of the two components of the surface, in this case coating (f1) and air (f2). θ1 is the contact angle if the droplet was purely in contact with the coating, and θ2 is the contact angle if the droplet was purely in contact with air. First we need to calculate the f1 and f2 using the data we are given for water. We have θCB = 148°, θ1 = 132°, and θ2 is taken to be 180° (contact angle on air). We then have cos(148) = f1.cos(132) + f2.cos(180). By using f1 + f2 = 1, we can solve to give f1 = 0.46 and f2 = 0.54. Next we need to calculate the contact angle of hexadecane on the rough surface, we have θ1 = 102°, f1 = 0.46, f2 = 0.54, and θ2 is taken to be 180° (contact angle on air). Therefore, θCB = 129° for hexadecane. The question however asks about a droplet of octane. Octane is a shorter oil molecule than hexadecane and has a lower surface tension than hexadecane. For a given surface, the contact angle of octane is therefore always lower than for hexadecane. Therefore, the answer is 124° as this is the only answer lower than the 129° of hexadecane.  The answer is (D)'''
    question_shot = f'''Question: {question}
Choices:
(A) {choice1}
(B) {choice2}
(C) {choice3}
(D) {choice4}
Thought Process: '''
    combined_prompt = shot1 + "\n\n" + shot2 + "\n\n" + shot3 + "\n\n" + shot4 + "\n\n" + shot5 + "\n\n" + question_shot
    return combined_prompt