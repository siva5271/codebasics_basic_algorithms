import pandas as pd
import numpy as np

x_raw=np.array([1,2,3,4,5])
y_raw=np.array([2*(x+1)+1 for x in x_raw])
no_of_iterations=10000
learning_rate=0.01
m_curr=b_curr=0
n=len(x_raw)

for i in range(no_of_iterations):
    mb=-(2/n)*sum(x_raw*(y_raw-((m_curr*x_raw)+b_curr)))
    bb=-(2/n)*sum(y_raw-((m_curr*x_raw)+b_curr))
    m_curr-=learning_rate*mb
    b_curr-=learning_rate*bb
    cost=sum((y_raw-((m_curr*x_raw)+b_curr))**2)/n
    print(f"M:{m_curr} B:{b_curr} cost:{cost}")
    