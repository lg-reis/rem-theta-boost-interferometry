"""
================================================================================
REM-θ-framework — Gravitational Quantum Interferometry
Parametrized simulation script — v9 (neutron COW only; side-panel layout;
  curve labels clarified)

DEVELOPMENT AND ATTRIBUTION:
  Theoretical Framework : Luís Gustavo Costa Reis
    - θ-parametrization of the Lorentz tensor and its geometric
      reformulation of gravitationally-induced quantum phase shift
    - Physical concepts and hypotheses
  Implementation         : Claude (Anthropic)
    - Computational translation of theory
    - Visualization design and figure generation
    - Numerical validation
  Article : Relativistic electromagnetic mass: applications to the
            Lorentz boost and gravitational quantum interferometry
            [DOI to be added after publication]

USAGE
-----
  Terminal:  python cow_simulation_final.py      (from any directory)
  Spyder:    run cow_simulation_final.py

OUTPUT (all paths relative to this script's folder)
------
  figures/cow_fig_neutron.png          neutron interferometry (COW effect):
                                       main plot + residuals (left), legend
                                       and fit numbers in a side panel (right)
  cow_results.txt                      numerical summary: q values, fit
                                       parameters (8, with bounds), free-q
                                       fit, residual diagnostics

SCOPE:
  Only the neutron (COW) case is simulated. Electrons and photons are not
  treated: the electron case would be a pure identity test with no data, and
  the geometric θ-form assumes m > 0, so photons need their own derivation. For the COW
  neutron, θ = arctan(λ_C/λ_dB) ~ 1.2e-5 rad: the regime is essentially
  non-relativistic, so the comparison checks the algebraic equivalence, not
  the relativistic structure in θ.

NEUTRON INPUT — specify ONE (priority: V > E > λ_dB):
  V_NEUTRON_M_S : velocity (m/s)         → λ_dB = h/(m·v)
  E_NEUTRON_meV : kinetic energy (meV)   → v = √(2E/m)
  LAM_NEUTRON_A : λ_dB directly (Å)     → fallback
  λ_C(neutron) = 1.3196 fm is a constant — NOT a user input.

PHYSICS
-------
  Standard QM:       Δφ = 2π m² g A λ_dB sin(δ) / h²      (non-relativistic
                     expression, e.g. Sakurai)
  REM-θ-framework:   Δφ = 2π g A cot(θ) sin(δ) / (c² λ_C),  θ=arctan(pc/E0)
                     (geometric θ-form of the article, Sec. 4)
  Identity:          cot(θ)/λ_C ≡ m²λ_dB/h²  (asserted at run time for the
                     neutron; residual ~1e-16).

NEUTRON FIT (Eq. 54 of Staudenmann et al., 8 free parameters, q FIXED at
  the theoretical value): A, B, C, D, Q, Δ1, φ0, β0 -- nuisance parameters of
  the background/contrast envelope, with no interpretation in terms of θ or
  of the framework. Their fitted values and bounds are written to
  cow_results.txt on every run (this is their only home; the article figure
  and caption report only the four diagnostic numbers below, not the 8
  parameters). The script also reports (i) a fit with q free, (ii) reduced
  chi² (Poisson errors), Durbin-Watson and lag-1 autocorrelation of the
  residuals, and (iii) the RMS/amplitude ratio, with amplitude ≡ half of
  (max − min) of the data. These numbers (RMS %, chi2_red, Durbin-Watson,
  q_free, theta) are printed directly on the figure's side panel.

NOTE: This is a proof-of-concept simulation. The algebraic identity above
is exact by construction, so agreement with standard QM is guaranteed
wherever standard QM is already experimentally validated -- this is not
an independent experimental test of the underlying framework.
================================================================================
"""

import os
import numpy as np
import matplotlib
from scipy.optimize import curve_fit

# ── Backend: always render to file, no live window ─────────────
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# ════════════════════════════════════════════════════════════════
# USER PARAMETERS — modify here
# ════════════════════════════════════════════════════════════════

# ── Neutron kinematic state (priority: V > E > λ_dB) ─────────
V_NEUTRON_M_S  = 3732.11  # velocity (m/s)         — 0 = ignore
E_NEUTRON_meV  = 0.0      # kinetic energy (meV)   — 0 = ignore
LAM_NEUTRON_A  = 0.0      # λ_dB fallback (Å)      — 0 = ignore
# All zero → default Staudenmann 1980: λ_dB = 1.060 Å

# ── Neutron interferometer geometry (Staudenmann 1980) ────────
A_SLAB_MM      = 2.464    # slab thickness a (mm)
D_SLAB_MM      = 34.518   # slab separation d (mm)
D_PLANES_A     = 1.920    # Si(220) plane spacing (Å)
C_BEND         = 34.57    # bending constant (rad·Å), Eq.50

# ── Experimental data for neutron panel ───────────────────────
DATA_FILE_N    = os.path.join(HERE, 'data', 'Staudenmann_Fig11.csv')  # phi_deg;counts
                           # '' = synthetic demo data

# ── Gravity ───────────────────────────────────────────────────
# Earth: 9.80665 | Moon: 1.62 | Mars: 3.72 | Jupiter: 24.79
G_LOCAL        = 9.80665  # m/s²

# ── Output ────────────────────────────────────────────────────
OUTPUT_FOLDER  = os.path.join(HERE, 'figures')
RESULTS_FILE   = os.path.join(HERE, 'cow_results.txt')
DPI            = 160
IDENT_TOL      = 1e-14    # tolerance asserted for the algebraic identity

# ════════════════════════════════════════════════════════════════
# CONSTANTS (CODATA 2018)
# ════════════════════════════════════════════════════════════════
h_pl = 6.62607015e-34
c_l  = 2.99792458e8
eV   = 1.602176634e-19
mn   = 1.67492749804e-27;  lCn = h_pl/(mn*c_l)   # 1.3196 fm

# ════════════════════════════════════════════════════════════════
# PHYSICS FUNCTIONS
# ════════════════════════════════════════════════════════════════
def phi_std(mass,lam,A,delta,g=None):
    g=g or G_LOCAL
    return 2*np.pi*mass**2*g*A*lam*np.sin(delta)/h_pl**2

def phi_fw(mass,lam,A,delta,lC,g=None):
    """Framework (geometric θ-form, Sec. 4): θ derived from λ_dB via θ=arctan(pc/E0)=arctan(λ_C/λ_dB);
    λ_C is the Compton wavelength of the species. Algebraically identical to
    the standard (non-relativistic) expression phi_std."""
    g=g or G_LOCAL
    th=np.arctan(h_pl/(lam*mass*c_l))
    return 2*np.pi*g*A*(np.cos(th)/np.sin(th))*np.sin(delta)/(c_l**2*lC)

# ════════════════════════════════════════════════════════════════
# NEUTRON SETUP
# ════════════════════════════════════════════════════════════════
if V_NEUTRON_M_S>0:
    v_n=V_NEUTRON_M_S; lam_n=h_pl/(mn*v_n); src='velocity'
elif E_NEUTRON_meV>0:
    v_n=np.sqrt(2*E_NEUTRON_meV*1e-3*eV/mn); lam_n=h_pl/(mn*v_n); src='energy'
elif LAM_NEUTRON_A>0:
    lam_n=LAM_NEUTRON_A*1e-10; v_n=h_pl/(mn*lam_n); src='lambda_dB'
else:
    lam_n=1.060e-10; v_n=h_pl/(mn*lam_n); src='default (1.060 Å)'
lam_n_A=lam_n*1e10; th_n=np.arctan(h_pl/(lam_n*mn*c_l))
E_kin=0.5*mn*v_n**2/eV*1000

a_c=A_SLAB_MM*1e-3; d_c=D_SLAB_MM*1e-3; d_pl=D_PLANES_A*1e-10
thB=np.arcsin(lam_n/(2*d_pl))
A_p=(2*d_c**2+2*a_c*d_c*np.cos(thB))*np.tan(thB)
q_grav=phi_std(mn,lam_n,A_p,np.pi/2)
q_fw_n=phi_fw(mn,lam_n,A_p,np.pi/2,lCn)
diff_n=abs(q_fw_n-q_grav)/q_grav
assert diff_n<IDENT_TOL, "Framework/canonical identity failed for neutron"
q_bend=(C_BEND/lam_n_A)*np.sin(thB)**2
q_tot=q_grav+q_bend

if DATA_FILE_N and os.path.isfile(DATA_FILE_N):
    rows=[]
    with open(DATA_FILE_N) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'):
                continue
            p=line.split(';')
            if len(p)==2:
                rows.append((float(p[0].replace(',','.')),
                             float(p[1].replace(',','.'))))
    dn=np.array(rows); phi_exp=dn[:,0]; cnt_exp=dn[:,1]
    idx=np.argsort(phi_exp); phi_exp=phi_exp[idx]; cnt_exp=cnt_exp[idx]
    dlbl='Experimental data (digitized)'
else:
    if DATA_FILE_N:
        print(f"[WARNING] Data file '{DATA_FILE_N}' not found -- "
              f"falling back to synthetic demo data. "
              f"Download the CSV from the repository to reproduce the "
              f"figure with the real Staudenmann et al. (1980) data.")
    np.random.seed(42)
    phi_exp=np.sort(np.random.uniform(-40,38,180))
    r_s=np.radians(phi_exp)
    cnt_exp=(2000-900*np.exp(-2.3*np.sin(r_s)**2)
             *np.cos(q_tot*np.sin(r_s))+np.random.normal(0,200,180))
    dlbl='Synthetic data (demo)'

# ── Fit model: Eq. 54 of Staudenmann et al. (1980) ─────────────
PARAM_NAMES=['A','B','C','D','Q','Delta1','phi0','beta0']
LB=[500,-20,-1,50,0.0,-np.pi,-10,-np.pi]
UB=[4000,20,1,2500,20.0,np.pi,10,np.pi]

def make_model(q):
    """I(phi) = A + B*phi + C*phi^2 + D*cos(Q*sin(phi)+Delta1)
                                      *cos(q*sin(phi-phi0)+beta0).
    q is FIXED from the physics (not a free parameter) in the main fit:
    the purpose is to test whether the theoretical q matches the data."""
    def model(pd,A,B,C,D,Q,Delta1,phi0,beta0):
        r=np.radians(pd); r0=np.radians(pd-phi0)
        bg=A+B*pd+C*pd**2
        mod=D*np.cos(Q*np.sin(r)+Delta1)
        return bg+mod*np.cos(q*np.sin(r0)+beta0)
    return model

def start_points(n_extra=12,seed=0):
    p0=[cnt_exp.mean(),0.,0.,(cnt_exp.max()-cnt_exp.min())/2,1.0,0.,0.,0.]
    rng=np.random.default_rng(seed)
    scale=np.array([100,1,.01,200,2,1,3,1])
    starts=[np.array(p0,dtype=float)]
    for _ in range(n_extra):
        s=np.array(p0)+rng.normal(size=8)*scale
        starts.append(np.clip(s,np.array(LB)+1e-6,np.array(UB)-1e-6))
    return starts

def fit_fixed_q(q):
    """Best of several seeded starts (the first is the original one)."""
    best=None
    for s in start_points():
        try:
            po,pc=curve_fit(make_model(q),phi_exp,cnt_exp,p0=s,
                            bounds=(LB,UB),maxfev=200000)
        except RuntimeError:
            continue
        rms=np.sqrt(np.mean((cnt_exp-make_model(q)(phi_exp,*po))**2))
        if best is None or rms<best[0]-1e-9:
            best=(rms,po,pc)
    return best

rms_S,popt_n,pcov_n=fit_fixed_q(q_tot)
A_f,B_f,C_f,D_f,Q_f,Delta1_f,phi0_f,beta0_f=popt_n
mdl_n=make_model(q_tot)

# ── Free-q fit (q as a 9th parameter), started from the fixed-q solution ──
def model_free(pd,A,B,C,D,Q,Delta1,phi0,beta0,q):
    return make_model(q)(pd,A,B,C,D,Q,Delta1,phi0,beta0)
try:
    popt_free,pcov_free=curve_fit(model_free,phi_exp,cnt_exp,
        p0=list(popt_n)+[q_tot],bounds=(LB+[20.0],UB+[45.0]),maxfev=200000)
    q_free=popt_free[-1]; q_free_err=float(np.sqrt(pcov_free[-1,-1]))
    rms_free=np.sqrt(np.mean((cnt_exp-model_free(phi_exp,*popt_free))**2))
except Exception:                              # pragma: no cover
    q_free=q_free_err=rms_free=float('nan')

# ── Fit without offset parameters (phi0 = beta0 = 0) ───────────
def model_no_off(pd,A,B,C,D,Q,Delta1):
    return make_model(q_tot)(pd,A,B,C,D,Q,Delta1,0.0,0.0)
try:
    p_no,_=curve_fit(model_no_off,phi_exp,cnt_exp,p0=list(popt_n[:6]),
                     bounds=(LB[:6],UB[:6]),maxfev=200000)
    rms_no_off=np.sqrt(np.mean((cnt_exp-model_no_off(phi_exp,*p_no))**2))
except Exception:                              # pragma: no cover
    rms_no_off=float('nan')

# ── Residual diagnostics (paper fit: q fixed) ──────────────────
resid=cnt_exp-mdl_n(phi_exp,*popt_n)
n_pts=len(phi_exp)
chi2_red=float(np.sum(resid**2/np.maximum(cnt_exp,1.0))/(n_pts-8))
dw=float(np.sum(np.diff(resid)**2)/np.sum(resid**2))
ac1=float(np.corrcoef(resid[:-1],resid[1:])[0,1])
amp_half=(cnt_exp.max()-cnt_exp.min())/2
rms_pct=100*rms_S/amp_half
at_bound=[nm for nm,v,lo,hi in zip(PARAM_NAMES,popt_n,LB,UB)
          if abs(v-lo)<1e-3*max(1,abs(lo)) or abs(v-hi)<1e-3*max(1,abs(hi))]

pf2=np.linspace(-41,40,3000)
def crv_c(pd,q):
    """Realistic curve: full modulation term (contrast reduction, as in data)."""
    r=np.radians(pd); r0=np.radians(pd-phi0_f)
    bg=A_f+B_f*pd+C_f*pd**2
    mod=D_f*np.cos(Q_f*np.sin(r)+Delta1_f)
    return bg+mod*np.cos(q*np.sin(r0)+beta0_f)
def crv_p(pd,q):
    """Idealized curve: constant amplitude D_f, no contrast-loss modulation."""
    r0=np.radians(pd-phi0_f)
    bg=A_f+B_f*pd+C_f*pd**2
    return bg+D_f*np.cos(q*np.sin(r0)+beta0_f)

c_pure=crv_p(pf2,q_fw_n)
c_fw  =crv_c(pf2,q_fw_n+q_bend)
c_st  =crv_c(pf2,q_grav+q_bend)
ymin=min(cnt_exp.min(),c_fw.min())-300
ymax=max(cnt_exp.max(),c_fw.max())+250

# ════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ════════════════════════════════════════════════════════════════
print("="*55)
print("REM-θ-framework — Gravitational Interferometry (COW)")
print("="*55)
print(f"\n[Neutron] {src}: λ_dB={lam_n_A:.4f}Å, v={v_n:.2f}m/s, E={E_kin:.3f}meV")
print(f"  λ_C(n)  = {lCn*1e15:.4f} fm  (Compton, constant)")
print(f"  q_grav  = {q_grav:.4f} rad  (std = fw, diff={diff_n:.2e})")
print(f"  q_bend  = {q_bend:.4f} rad")
print(f"  q_total = {q_tot:.4f} rad, RMS={rms_S:.0f} cts ({rms_pct:.1f}% of half range)")
print(f"  q_free  = {q_free:.3f} ± {q_free_err:.3f} rad (statistical only), RMS={rms_free:.0f} cts")
print(f"  residuals: reduced chi2 (Poisson) = {chi2_red:.2f}, Durbin-Watson = {dw:.2f}, lag-1 autocorr = {ac1:.2f}")
print(f"  theta = arctan(lambda_C/lambda_dB) = {th_n:.3e} rad (essentially non-relativistic regime)")

# ════════════════════════════════════════════════════════════════
# FIGURE: main + residuals (left) and a legend/numbers side panel (right)
# ════════════════════════════════════════════════════════════════
os.makedirs(OUTPUT_FOLDER,exist_ok=True)

TITLE_TOP="Gravity-Induced Quantum Phase Shift — REM-$\\theta$-framework"
EQ_LINE=(r"$\Delta\phi_{\rm geom}=\frac{2\pi gA\cot\theta\sin\delta}{c^2\lambda_C}"
         r"\equiv\frac{2\pi m^2gA\lambda_{\rm dB}\sin\delta}{h^2}$"
         r"  (error $<10^{-14}$)")

fig=plt.figure(figsize=(12,6.7))
fig.text(0.5,0.975,TITLE_TOP,ha='center',va='top',fontsize=12,fontweight='bold')
fig.text(0.5,0.925,EQ_LINE,ha='center',va='top',fontsize=11)
fig.text(0.5,0.845,
         r'Neutron — Staudenmann et al. (1980), '
         rf'$\lambda_{{\rm dB}}={lam_n_A:.3f}\,\AA$',
         ha='center',va='top',fontsize=10)

LEFT, RIGHT_MAIN, RIGHT = 0.075, 0.675, 0.985
TOP, BOTTOM = 0.775, 0.10

gs_left = GridSpec(2,1,figure=fig,height_ratios=[3,1],hspace=0.07,
                   left=LEFT,right=RIGHT_MAIN,top=TOP,bottom=BOTTOM)
ax   = fig.add_subplot(gs_left[0])
ax_r = fig.add_subplot(gs_left[1])

gs_side = GridSpec(1,1,figure=fig,left=0.735,right=RIGHT,top=TOP,bottom=BOTTOM)
ax_s = fig.add_subplot(gs_side[0])
ax_s.axis('off')

# ---- main plot (no in-plot legend / info box: both move to the side panel)
l_pu,=ax.plot(pf2,c_pure,'g:',lw=1.5,zorder=2,alpha=0.85)
l_fw,=ax.plot(pf2,c_fw,  'b-',lw=2.0,zorder=3)
l_st,=ax.plot(pf2,c_st,  'r--',lw=1.4,zorder=4,alpha=0.7)
l_dt =ax.scatter(phi_exp,cnt_exp,s=7,c='black',zorder=6,alpha=0.85)
ax.set_xlim(-41,40); ax.set_ylim(ymin,ymax)
ax.set_ylabel(r'Neutron counts $C_3$',fontsize=11)
ax.grid(True,alpha=0.2)
ax.xaxis.set_ticklabels([])
ax.tick_params(axis='both',labelsize=9)

# ---- residuals
ax_r.scatter(phi_exp,resid,s=9,c='darkred',alpha=0.8,zorder=5)
ax_r.axhline(0,color='k',lw=1)
ax_r.axhspan(-rms_S,rms_S,color='gray',alpha=0.15)
ax_r.set_xlim(-41,40)
ax_r.set_xlabel(r'Rotation angle $\phi$ (deg)',fontsize=10.5)
ax_r.set_ylabel('Resid.',fontsize=9.5)
ax_r.tick_params(axis='both',labelsize=8.5)
ax_r.grid(True,alpha=0.2)

# ---- side panel: ONE white box, two sections (bold titles, left-aligned)
ax_s.set_xlim(0,1); ax_s.set_ylim(0,1)

panel = FancyBboxPatch((0.02,0.02), 0.96, 0.96,
                        boxstyle='round,pad=0.015,rounding_size=0.02',
                        transform=ax_s.transAxes,
                        facecolor='white', edgecolor='gray',
                        linewidth=0.9, zorder=1)
ax_s.add_patch(panel)

X0, XSW0, XSW1, XLBL = 0.07, 0.07, 0.19, 0.22  # swatch span + label start

curves = [
    dict(color='black', marker='o', ls='None',
         label='Experimental data'),
    dict(color='r', marker=None, ls='--', lw=1.4,
         label='Standard form + bending corr.\n(identical to $\\theta$-form)'),
    dict(color='g', marker=None, ls=':', lw=1.6,
         label=f'$q_{{\\rm grav}}$ only: $q={q_fw_n:.1f}$ rad\n(no bending corr., ideal contrast)'),
    dict(color='b', marker=None, ls='-', lw=2.1,
         label=f'$\\theta$-form + bending corr.:\n$q={q_fw_n+q_bend:.1f}$ rad'),
]

y = 0.93
ax_s.text(X0, y, r'$\mathbf{Curves}$', transform=ax_s.transAxes,
          fontsize=9.3, ha='left', va='top', zorder=2)
y -= 0.075

for cv in curves:
    ym = y - 0.012
    if cv['marker'] == 'o':
        ax_s.plot([(XSW0+XSW1)/2], [ym], marker='o', ls='None',
                   color=cv['color'], markersize=5,
                   transform=ax_s.transAxes, zorder=2, clip_on=False)
    else:
        ax_s.plot([XSW0,XSW1], [ym,ym], color=cv['color'], ls=cv['ls'],
                   lw=cv['lw'], transform=ax_s.transAxes, zorder=2,
                   clip_on=False)
    ax_s.text(XLBL, y, cv['label'], transform=ax_s.transAxes,
              fontsize=8.3, ha='left', va='top', zorder=2, linespacing=1.25)
    y -= 0.075 + 0.04 * cv['label'].count('\n')

y -= 0.015
ax_s.text(X0, y, r'$\mathbf{Fit\ results}$', transform=ax_s.transAxes,
          fontsize=9.3, ha='left', va='top', zorder=2)
y -= 0.075

results_lines = [
    rf'$q_{{\rm grav}}={q_grav:.1f}$ rad',
    rf'$q_{{\rm bend}}={q_bend:.2f}$ rad',
    rf'RMS$={rms_S:.0f}$ cts (${rms_pct:.1f}\%$)',
    rf'$q_{{\rm free}}={q_free:.2f}\pm{q_free_err:.2f}$ rad',
    rf'$\theta={th_n*1e5:.1f}\times10^{{-5}}$ rad',
    rf'$\chi^2_{{\rm red}}\approx{chi2_red:.1f}$,  DW$={dw:.2f}$',
]
for line in results_lines:
    ax_s.text(X0, y, line, transform=ax_s.transAxes,
              fontsize=8.3, ha='left', va='top', zorder=2)
    y -= 0.058

fig_path = os.path.join(OUTPUT_FOLDER,'cow_fig_neutron.png')
plt.savefig(fig_path,dpi=DPI,bbox_inches='tight')
plt.close()
print(f"\nSaved: {fig_path}")

# ════════════════════════════════════════════════════════════════
# NUMERICAL SUMMARY
# ════════════════════════════════════════════════════════════════
with open(RESULTS_FILE,'w',encoding='utf-8') as f:
    f.write("REM-θ-framework — Results\n"+"="*40+"\n")
    f.write(f"G_LOCAL = {G_LOCAL} m/s²\n\n")
    f.write(f"[Neutron] {src}: λ_dB={lam_n_A:.4f}Å  (data: {dlbl})\n")
    f.write(f"  λ_C(n)  = {lCn*1e15:.4f} fm\n")
    f.write(f"  q_grav  = {q_grav:.6f} rad\n")
    f.write(f"  q_fw    = {q_fw_n:.6f} rad\n")
    f.write(f"  |diff|  = {diff_n:.2e}\n")
    f.write(f"  q_bend  = {q_bend:.6f} rad\n")
    f.write(f"  q_total = {q_tot:.6f} rad\n")
    f.write(f"  RMS     = {rms_S:.1f} counts  ({rms_pct:.1f}% of half range "
            f"(max-min)/2 = {amp_half:.1f})\n\n")
    f.write("  Fit with q FIXED at q_total (8 free parameters, N = %d points)\n"%n_pts)
    f.write("  These 8 parameters are nuisance parameters of the background/contrast\n")
    f.write("  envelope, with no interpretation in terms of theta; reported only here.\n")
    f.write("    %-8s %12s %12s %12s\n"%("param","value","lower","upper"))
    for nm,v,lo,hi in zip(PARAM_NAMES,popt_n,LB,UB):
        f.write("    %-8s %12.5g %12.5g %12.5g\n"%(nm,v,lo,hi))
    f.write(f"    parameters at a bound: {at_bound if at_bound else 'none'}\n")
    f.write(f"    fit without offsets (phi0 = beta0 = 0, 6 free): RMS = {rms_no_off:.1f} counts\n\n")
    f.write("  Fit with q FREE (9 parameters)\n")
    f.write(f"    q_free = {q_free:.4f} ± {q_free_err:.4f} rad (statistical only; residuals are correlated)\n")
    f.write(f"    RMS    = {rms_free:.1f} counts\n")
    f.write(f"    (q_total - q_free)/q_total = {(q_tot-q_free)/q_tot*100:.2f} %\n\n")
    f.write("  Residual diagnostics (q fixed)\n")
    f.write(f"    reduced chi2 (Poisson errors, N-8 dof) = {chi2_red:.2f}\n")
    f.write(f"    Durbin-Watson = {dw:.2f}   lag-1 autocorrelation = {ac1:.2f}\n")
    f.write("    (independent noise would give DW ~ 2 and autocorrelation ~ 0)\n\n")
    f.write(f"  theta = arctan(lambda_C/lambda_dB) = {th_n:.3e} rad (essentially non-relativistic regime)\n")
print(f"Saved: {RESULTS_FILE}")
print(f"All figures in: {OUTPUT_FOLDER}")
print("Done.")
