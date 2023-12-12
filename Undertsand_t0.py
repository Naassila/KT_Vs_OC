import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
def lognpdf(x, mu, sigma):
    x = np.where(x <= 0, np.inf, x )
    y = np.exp(-0.5*((np.log(x) - mu)/sigma)**2) / (x * np.sqrt(2*np.pi) * sigma)
    return y

t0 = 0
D = 1
sigma = 0.25
mu = -2

t0s = [0, 0.1, 0.2]
data = pd.DataFrame(data=np.zeros((1000, 4)), columns=['time']+t0s)

Nt = 1000
T = 0.8
t = np.linspace(0, T, Nt)
data.time=t
for t0 in t0s:
    tt0 = t-t0
    v = lognpdf(tt0, mu, sigma)
    data[t0] = v

df = data.melt('time')
sns.lineplot(data=df, x='time', y='value', hue='variable')
plt.savefig('t0_understand.svg')

print('yep')