import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, least_squares
from qpsolvers import solve_qp
from qpsolvers.problem import Problem
from qpsolvers.solve_problem import solve_problem

def lognpdf(x, mu, sigma, t0):
    x = np.where(x <= 0, np.inf, x )
    y = np.exp(-0.5*((np.log(x-t0) - mu)/sigma)**2) / ((x-t0) * np.sqrt(2*np.pi) * sigma)
    return y

def func(x, dti, dtj, ai, aj):
    a = dti/dtj - (1-np.exp(-x*np.sqrt(2*np.log(ai)))) / (1-np.exp(-x*np.sqrt(2*np.log(aj))))
    return a

def func_sigma(x, tm, tmin, tmax):
    a = (tmax-tmin)*(np.exp(-x**2)-np.exp(-3*x))/(2*np.sinh(3*x)) - (tm-tmin)
    return a

def home_fit_lognpdf(t , f, ti, tj, u_bounds, s_bounds):
    # based on https://journal.hep.com.cn/fcs/EN/article/downloadArticleFile.do?attachType=PDF&id=6433
    abf = np.abs(f)
    t_max = t[abf.argmax()]
    finter = interp1d(t, abf, kind='cubic')
    dti = t_max - t_max * ti
    dtj = t_max - t_max * tj
    ai = abf.max()/finter(ti*t_max)
    aj = abf.max()/finter(tj*t_max)

    f = lambda x: func(x, dti, dtj, ai, aj)
    # sigma = fsolve(f, s_bounds[-1])
    res = least_squares(f, s_bounds[-1], bounds=(s_bounds[0], s_bounds[1]))
    sigma = res.x
    mu = np.log(dti / (np.exp(sigma**2)*(1-np.exp(-sigma*np.sqrt(2*np.log(ai))))))

    D = abf.max()*np.sqrt(2*np.pi)*sigma/np.exp(-mu+0.5*sigma**2)

    t0 = t_max - np.exp(mu-sigma**2)

    return mu, sigma, D, t0

def inflexion_points_logparam(t, f, df, s_bounds):
    # based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4668345
    abf = np.abs(f)
    df = - df
    t_m = t[abf.argmax()]
    i1 = df.argmax()
    tinf1 = t[i1]
    i2 = df.argmin()
    tinf2 = t[i2]
    v1_100 = abf-abf/100
    imin = v1_100.argmin()
    if t[imin]<t_m:
        imax = v1_100[abf.argmax():].argmin() + abf.argmax()
    else:
        imax = imin
        imin = v1_100[:abf.argmax()].argmin()
    tmax = t[imax]
    tmin = t[imin]

    f = lambda x: func_sigma(x, t_m, tmin, tmax)
    res = least_squares(f, s_bounds[-1], bounds=(s_bounds[0], s_bounds[1]))
    sigma = res.x

    alpha1 = np.exp(-sigma * (sigma + np.sqrt(sigma ** 2 + 4)) / 2)
    alpha2 = np.exp(-sigma * (sigma - np.sqrt(sigma ** 2 + 4)) / 2)
    mu = sigma**2 + np.log((tinf2-tinf1)/(alpha2-alpha1))

    t0 = t_m-np.exp(mu-sigma**2)
    D = abf.max()*sigma*np.sqrt(2*np.pi)*np.exp(mu-0.5*sigma**2)

    return mu, sigma, D, t0

class set_problem:
    def __init__(self, qLow ,qUpp, dqMax, tNode, qNode, nGrid):
        self.qLow = qLow
        self.qUpp = qUpp
        self.dqMax = dqMax
        self.tNode = tNode
        self.qNode = qNode
        self.nGrid = nGrid

    def chebyInterpolate(self, f, t_inter, t_bounds):
        if t_inter[0]<t_bounds[0] or t_inter[-1]>t_bounds[-1]:
            raise ValueError('the interpolation may yield wrong results')

        if len(f.shape) == 1:
            k = 1
            n = f.shape[0]
            f = f.reshape((1, f.shape[0]))
            t_inter = t_inter.reshape((1, t_inter.shape[0]))
            t_bounds = np.array(t_bounds)
            t_bounds = t_bounds.reshape((1, t_bounds.shape[0]))
        else:
            raise ValueError('not coded yet')

        x_c, w_c = self.chebyPoints(n, t_bounds)
        o1 = np.ones(k)
        o2 = np.ones(t_inter.shape[1])

        num = np.zeros((k, t_inter.shape[1]))
        den = np.zeros((k, t_inter.shape[1]))
        for i in range(n):
            val = o1 * 1/(t_inter - x_c[:,i])
            if np.mod(i, 2) == 0:
                val=-val

            if i in [0, n-1]:
                num = num + 0.5 * (f[:,i] * o2) * val
                den = den + 0.5 * val
            else:
                num = num + f[:,i] * o2 *val
                den = den + val
        y = num / den

        # Correct values close to nodes
        nanIdx = np.isnan(y)
        if np.sum(nanIdx) > 0:
            finterp = interp1d(x=x_c[0], y=f, kind='cubic')
            y[nanIdx] = finterp(t_inter[nanIdx])[0]



        return y

    def chebyPoints(self, n, d):
        if n==1:
            x = 0
            weights = 2
        else:
            m = n-1
            x = np.sin(np.pi*(np.arange(-m, m+2, 2))/(2*m))
            x = (np.diff(d)*x + np.sum(d))/2
            mod_python = np.round(m/2, decimals=3)*2 - m
            u0 = 1/(m**2-1+mod_python)
            L = np.arange(0, m, 1)
            r = 2/(1-4*np.array([np.min([iL, m-iL]) for iL in L])**2)
            weights = np.hstack([np.fft.ifft(r-u0).real,u0])
            weights = weights*np.diff(d)/2
        return x, weights

    def chebyDerivative(self, f, d, D, DD, DDD):
        Df = np.matmul(D, f.T).T
        DDf = np.matmul(DD, f.T).T
        DDDf = np.matmul(DDD, f.T).T
        return Df, DDf, DDDf

    def chebyDiffMatrix(self, n, d = [-1, 1]):
        if len(np.array(d).shape) == 1:
            d = np.array(d)
            d = d.reshape((1, d.shape[0]))

        x, weights = self.chebyPoints(n, d)

        w = np.ones(n)
        w[1:-1:2] = -1
        w[0] /= 2
        w[-1] /= 2

        w_matrix = np.outer(1/w, w)

        X = np.zeros((n,n))
        for ix in range(n):
            idx = np.arange(ix+1, n, 1)
            X[ix, idx] = 1/(x[0,ix] - x[0,idx])

        X = X - X.T
        D = w_matrix*X
        D = D - np.diag(np.sum(D, 1))

        return D

    def chebysegment(self, icost, n, d, alpha_beta):
        D = self.chebyDiffMatrix(n, d)
        DD = np.matmul(D, D) # qddot
        DDD = np.matmul(DD, D) # jerk

        t,w = self.chebyPoints(n, d)

        A = np.vstack((D, -D))
        b = self.dqMax*np.ones(2*n)

        if icost == 'Minimize jerk':
            W = np.diag(w)
            H = 0.5 * np.matmul(np.transpose(DDD), np.matmul(W, DDD))
        elif icost == 'Minimize jerk and time':
            W = np.diag(w) * np.diag(t**2)
            H = 0.5 * np.matmul(np.transpose(DDD), np.matmul(W, DDD))
        elif icost == 'Minimize jerk, energy and time':
            W = np.diag(w) * np.diag(t**2)
            H = 0.5 * (np.matmul(np.transpose(DDD), np.matmul(W, DDD))+
                       alpha_beta*np.matmul(np.transpose(D), np.matmul(W, D)))
        else:
            raise ValueError('Unknown cost function')

        return [A, b, H, t, D, DD, DDD]

    def solve(self, icost, n, d, alpha_beta):
        nDecVar = np.sum(self.nGrid)
        nSegment = len(self.nGrid)

        for i in range(nSegment):
            # todo: multiple segments
            #  Sol_all.append(self.chebysegment(icost, n, d, alpha_beta))
            A, b, H, t, D, DD, DDD = self.chebysegment(icost, n, d, alpha_beta)
        nCstBc = 2*nSegment+2*(nSegment+1)

        Aeq = np.zeros((nCstBc, nDecVar))
        beq = np.zeros(nCstBc)
        finalIdx = np.cumsum(self.nGrid)
        startIdx = [0] #+ [0, finalIdx[1:-1]]

        cstIdx = -1
        for iseg in range(nSegment):
            cstIdx = cstIdx + 1
            Aeq[cstIdx, startIdx[iseg]] = 1
            beq[cstIdx] = self.qNode[iseg]

        for iseg in range(nSegment):
            cstIdx = cstIdx + 1
            Aeq[cstIdx, finalIdx[iseg]-1] = 1
            beq[cstIdx] = self.qNode[iseg+1]

        cstIdx = cstIdx + 1
        Aeq[cstIdx, :] = D[0, :] # null initial velocity
        cstIdx = cstIdx + 1
        Aeq[cstIdx,:] = D[-1, :] # null final velocity

        # constraints on rate at segment boundaries
        for i in range(nSegment-1):
            cstIdx = cstIdx + 1
            Aeq[cstIdx, :] = D[startIdx[i+1], :] - D[finalIdx[i], :]

        for i in range(nSegment-1):
            cstIdx = cstIdx + 1
            Aeq[cstIdx,:] = DD[startIdx[i + 1],:] - DD[finalIdx[i],:]

        # specify options and solve
        to_min = Problem(P = (H+np.transpose(H))/2,
                     q = np.zeros(nDecVar),
                     G= A,
                     h= b,
                     A= Aeq,
                     b= beq,
                     lb= self.qLow*np.ones(nDecVar),
                     ub= self.qUpp*np.ones(nDecVar),)
        solution = solve_problem(to_min,
                                 solver='scs', #,'ecos'
                                 )

        if solution.found:
            sol = solution.x
            cost = solution.obj
            for iseg in range(nSegment):
                tSpan = [t[0], t[-1]]
                grid_q = sol[startIdx[iseg]:finalIdx[iseg]]
                grid_q_derivatives = self.chebyDerivative(grid_q, tSpan, D, DD, DDD)
                grid_dq = grid_q_derivatives[0]
                t_interp = np.linspace(t[0], t[-1], 101)
                q = self.chebyInterpolate(grid_q, t_interp, tSpan)
                dq = self.chebyInterpolate(grid_dq, t_interp, tSpan)
                ddq = self.chebyInterpolate(grid_q_derivatives[1], t_interp, tSpan)
                dddq = self.chebyInterpolate(grid_q_derivatives[2], t_interp, tSpan)

        else:
            print('No convergence')

        return [t, grid_q, grid_dq, t_interp, q, dq, ddq, dddq, cost]