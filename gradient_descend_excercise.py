import pandas as pd
import numpy as np
import math

df=pd.read_csv("/home/siva/Downloads/py-master/ML/3_gradient_descent/Exercise/test_scores.csv")
x=np.array(df['math'])
y=np.array(df['cs'])
m_curr=b_curr=0
learning_rate=0.001
n=len(x)
print(x)
i=0
while i<1000:
    cost=sum((y-(m_curr*x)+b_curr)**2)/n
    print(f'M:{m_curr} B:{b_curr} Cost: {cost}')
    mb=-(2/n)*sum(x*(y-((m_curr*x)+b_curr)))
    bb=-(2/n)*sum(y-((m_curr*x)+b_curr))
    # if math.isclose(m_curr, m_curr-learning_rate*mb,  rel_tol=1e-20, abs_tol=0.0) and \
    #     math.isclose(b_curr, b_curr-learning_rate*bb,  rel_tol=1e-20, abs_tol=0.0):
    #     break
    m_curr-=learning_rate*mb
    b_curr-=learning_rate*bb
    i+=1
    
