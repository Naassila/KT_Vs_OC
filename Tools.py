import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, least_squares
from scipy.linalg import block_diag
# from qpsolvers import solve_qp
# from qpsolvers.problem import Problem
# from qpsolvers.solve_problem import solve_problem
# from casadi import *

def lognpdf(x, mu, sigma, t0):
    xt0 = x-t0
    xt0 = np.where(xt0 <= 0, np.inf, xt0 )
    y = np.exp(-0.5*((np.log(xt0) - mu)/sigma)**2) / ((xt0) * np.sqrt(2*np.pi) * sigma)
    return y

def func(x, dti, dtj, ai, aj):
    a = dti/dtj - (1-np.exp(-x*np.sqrt(2*np.log(ai)))) / (1-np.exp(-x*np.sqrt(2*np.log(aj))))
    return a

def func_sigma(x, tm, tmin, tmax):
    a = (tmax-tmin)*(np.exp(-x**2)-np.exp(-3*x))/(2*np.sinh(3*x)) - (tm-tmin)
    return a

def func_sigma_t0(x, t0, tm, tinf2, tinf1):
    alpha1 = np.exp(-x * (x + np.sqrt(x ** 2 + 4)) / 2)
    alpha2 = np.exp(-x * (x - np.sqrt(x ** 2 + 4)) / 2)
    a = (tinf2-tinf1)/(tm-t0)-(alpha2-alpha1)
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
    df = df # was -df
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

def detect_onset(x, threshold, n_above = 1, n_below= 0,  threshold2= None, n_above2= 1):

    if x.ndim != 1:
        raise ValueError(
            f"detect_onset works only for one-dimensional vector. You have {x.ndim} dimensions."
        )
    x = np.atleast_1d(x.copy())
    x[np.isnan(x)] = -np.inf
    inds = np.nonzero(x >= threshold)[0]
    if inds.size:
        # initial and final indexes of almost continuous data
        inds = np.vstack(
            (
                inds[np.diff(np.hstack((-np.inf, inds))) > n_below + 1],
                inds[np.diff(np.hstack((inds, np.inf))) > n_below + 1],
            )
        ).T
        # indexes of almost continuous data longer than or equal to n_above
        inds = inds[inds[:, 1] - inds[:, 0] >= n_above - 1, :]
        # minimum amplitude of n_above2 values in x to detect
        if threshold2 is not None and inds.size:
            idel = np.ones(inds.shape[0], dtype=bool)
            for i in range(inds.shape[0]):
                if (
                    np.count_nonzero(x[inds[i, 0] : inds[i, 1] + 1] >= threshold2)
                    < n_above2
                ):
                    idel[i] = False
            inds = inds[idel, :]
    if not inds.size:
        inds = np.array([])
    return inds
def inflexion_points_logparam_robust(t, f, df, s_bounds, plot_check=False):
    # based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4668345
    # Basically, we need to identify the following events:
    # t_m: occurence of maximum velocity
    # tinf1: occurence of maximum acceleration
    # tinf2: occurence of minimum acceleration
    # tmin: min bound of the interval
    # tmax max bound of the interval
    ab_v = np.abs(f)
    df = df # was -df
    onsets = detect_onset(ab_v, ab_v.max()*1/100)[0]
    tmin = t[onsets[0]]
    tmax = t[onsets[1]]
    iv_max = ab_v.argmax()
    t_m = t[iv_max]
    i1 = np.nanargmax(df)
    tinf1 = t[i1]
    i2 = np.nanargmin(df)
    tinf2 = t[i2]
    v_0 = ab_v.max()*1/100 # 1% max velocity to identify start and end of lognormal
    # vmin = (ab_v-v_0).argmin()
    # sort_indices = np.argsort(ab_v - ab_v.max() * 1 / 100)
    # imin_max = np.argsort(np.abs(np.diff(sort_indices)))[::-1][:2]
    # if iv_max<imin_max[1] and iv_max>imin_max[0]:
    #     tmax = imin_max[1]
    #     tmin = imin_max[0]
    # else:
    #     ordered_indices = np.argsort(np.abs(np.diff(sort_indices)))[::-1]
    #     tmin = t[ordered_indices[ordered_indices<iv_max][0]]
    #     tmax = t[ordered_indices[ordered_indices>iv_max][0]]

    if plot_check:
        plt.plot(t, f, label='Velocity')
        plt.plot(t, ab_v-v_0)
        # plt.plot(t, df, label='Acceleration')
        for it in [t_m, tmin, tmax, tinf1, tinf2]:
            plt.axvline(x = it)

    f = lambda x: func_sigma(x, t_m, tmin, tmax)
    res = least_squares(f, s_bounds[-1], bounds=(s_bounds[0], s_bounds[1]))
    sigma = res.x

    alpha1 = np.exp(-sigma * (sigma + np.sqrt(sigma ** 2 + 4)) / 2)
    alpha2 = np.exp(-sigma * (sigma - np.sqrt(sigma ** 2 + 4)) / 2)
    mu = sigma**2 + np.log((np.abs(tinf2-tinf1))/(alpha2-alpha1))
    #TODO: confirm abs value

    t0 = t_m-np.exp(mu-sigma**2)
    D = ab_v.max()*sigma*np.sqrt(2*np.pi)*np.exp(mu-0.5*sigma**2)

    return mu, sigma, D, t0, [t_m,tinf1,tinf2,tmin,tmax]

def inflexion_points_logparam_robust_t0_fixed(t, f, df, s_bounds, t0=0, plot_check=False):
    # based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4668345
    # Basically, we need to identify the following events:
    # t_m: occurence of maximum velocity
    # tinf1: occurence of maximum acceleration
    # tinf2: occurence of minimum acceleration
    # tmin: min bound of the interval
    # tmax max bound of the interval
    ab_v = np.abs(f)
    df = df # was -df
    onsets = detect_onset(ab_v, ab_v.max()*1/100)[0]
    tmin = t[onsets[0]]
    tmax = t[onsets[1]]
    iv_max = ab_v.argmax()
    t_m = t[iv_max]
    i1 = np.nanargmax(df)
    tinf1 = t[i1]
    i2 = np.nanargmin(df)
    tinf2 = t[i2]
    v_0 = ab_v.max()*1/100 # 1% max velocity to identify start and end of lognormal
    # vmin = (ab_v-v_0).argmin()
    # sort_indices = np.argsort(ab_v - ab_v.max() * 1 / 100)
    # imin_max = np.argsort(np.abs(np.diff(sort_indices)))[::-1][:2]
    # if iv_max<imin_max[1] and iv_max>imin_max[0]:
    #     tmax = imin_max[1]
    #     tmin = imin_max[0]
    # else:
    #     ordered_indices = np.argsort(np.abs(np.diff(sort_indices)))[::-1]
    #     tmin = t[ordered_indices[ordered_indices<iv_max][0]]
    #     tmax = t[ordered_indices[ordered_indices>iv_max][0]]

    if plot_check:
        plt.plot(t, f, label='Velocity')
        plt.plot(t, ab_v-v_0)
        # plt.plot(t, df, label='Acceleration')
        for it in [t_m, tmin, tmax, tinf1, tinf2]:
            plt.axvline(x = it)

    f = lambda x: func_sigma_t0(x, t0, t_m, tinf2, tinf1)
    # f = lambda x: func_sigma(x, t_m, tinf2, tinf1)
    res = least_squares(f, s_bounds[-1], bounds=(s_bounds[0], s_bounds[1]))
    sigma = res.x

    mu = sigma**2 + np.log(t_m-t0)
    D = ab_v.max()*sigma*np.sqrt(2*np.pi)*np.exp(mu-0.5*sigma**2)

    return mu, sigma, D, [t0], [t_m,tinf1,tinf2,tmin,tmax]

def snr(v,ln):
    error = ln-v
    signal_pow = np.linalg.norm(ln.flatten()) ** 2
    noise_pow = np.linalg.norm(error.flatten()) ** 2
    r = 10 * np.log10(signal_pow / noise_pow)
    return r
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

    def solve(self, icost, n, d, alpha_beta, second_joint=0):
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
                if second_joint ==1:
                    t_interp = np.linspace(0, t[-1], 101)
                    t_inter_crop = t_interp[np.where(t_interp==np.round(t[0], 3))[0][0]:]
                    q = np.hstack(
                        (np.ones((1,t_interp.shape[0]-t_inter_crop.shape[0]))*self.qNode[0],
                         self.chebyInterpolate(grid_q, t_inter_crop, tSpan)
                         ))
                    dq = np.hstack(
                        (np.zeros((1,t_interp.shape[0]-t_inter_crop.shape[0])),
                         self.chebyInterpolate(grid_dq, t_inter_crop, tSpan)
                         ))
                    ddq = np.hstack(
                        (np.zeros((1,t_interp.shape[0]-t_inter_crop.shape[0])),
                         self.chebyInterpolate(grid_q_derivatives[1], t_inter_crop, tSpan)
                         ))
                    dddq = np.hstack(
                        (np.zeros((1,t_interp.shape[0]-t_inter_crop.shape[0])),
                         self.chebyInterpolate(grid_q_derivatives[2], t_inter_crop, tSpan)
                         ))
                else:
                    t_interp = np.linspace(t[0], t[-1], 101)
                    q = self.chebyInterpolate(grid_q, t_interp, tSpan)
                    dq = self.chebyInterpolate(grid_dq, t_interp, tSpan)
                    ddq = self.chebyInterpolate(grid_q_derivatives[1], t_interp, tSpan)
                    dddq = self.chebyInterpolate(grid_q_derivatives[2], t_interp, tSpan)

        else:
            print('No convergence')

        return [t, grid_q, grid_dq, t_interp, q, dq, ddq, dddq, cost]

class set_problem_ocp:
    def __init__(self, qLow ,qUpp, dqMax, tNode, qNode, nGrid):
        self.qLow = qLow
        self.qUpp = qUpp
        self.dqMax = dqMax
        self.tNode = tNode
        self.qNode = qNode
        nGrid = 100
        self.nGrid = nGrid
        N = self.nGrid
        self.tf = MX.sym('tf')
        self.x = MX.sym("x", 2)  # x[0] = theta1,
        self.u = MX.sym("u")
        self.dN = self.tf/N

    def solve(self, alpha_beta):
        N = self.nGrid
        tf = MX.sym('tf')
        # tf=1.0
        self.dN = tf / N
        x = MX.sym("x", 2)  # x[0] = p, x[1] = v
        u = MX.sym("u")  # u[0] = a

        def dyn_fun(x, u):
            q = x[0]
            qdot = x[1]
            qddot = u
            return vertcat(qdot, qddot)

        def int_fun(xk, uk):
            M = 10
            state = xk
            control = uk
            for i in range(M):
                dstate = dyn_fun(state, control)
                state += dstate * self.dN / M
            return state

        # Formulate discrete time dynamics

        ### REGULAR PROBLEM
        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # Initialize
        w += [tf]
        lbw += [0]
        ubw += [1]
        w0 += [0.8]

        Xk = MX.sym('X0', 2)
        w += [Xk]
        lbw += [self.qNode[0], 0.0]
        ubw += [self.qNode[0], 0.0]
        w0 += [self.qNode[0], 0.0]
        # Formulate the NLP
        for k in range(N):
            # New NLP variable for the control
            Uk = MX.sym('U_' + str(k))
            J += (Uk**2 + 1000 * Xk[1]**2) * (k*self.dN)**4
            w += [Uk]
            lbw += [-1000]
            ubw += [1000]
            w0 += [0]

            # Integrate till the end of the interval
            Fk = int_fun(Xk, Uk)
            Xk_end = Fk

            # New NLP variable for state at end of interval
            Xk = MX.sym('X_' + str(k + 1), 2)
            if k + 1 == N:
                w += [Xk]
                lbw += [self.qNode[1], 0.0]
                ubw += [self.qNode[1], 0.0]
                w0 += [0] * 2
            else:
                w += [Xk]
                lbw += [self.qLow, -self.dqMax]
                ubw += [self.qUpp, self.dqMax]
                w0 += [0] * 2
            # Add equality constraint
            g += [Xk_end - Xk]
            ubg += [0] * 2
            lbg += [0] * 2



        # Create NLP solver
        opts = {'ipopt.linear_solver': 'mumps', 'ipopt.tol': 1e-6, 'ipopt.constr_viol_tol': 1e-6,
                'ipopt.hessian_approximation': 'limited-memory'}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        tf_opt = w_opt[0]
        p_opt = w_opt[1::3]
        v_opt = w_opt[2::3]
        u_opt = w_opt[3::3]

        import matplotlib.pyplot as plt
        T = np.linspace(0, tf_opt, p_opt.shape[0])
        fig, axes = plt.subplots(nrows=2, ncols=1, sharey=False)
        axes[0].plot(T,np.rad2deg(p_opt))
        axes[1].plot(T,v_opt)
        plt.figure()
        plt.plot(T[:-1], u_opt)
        plt.plot(u_opt)


