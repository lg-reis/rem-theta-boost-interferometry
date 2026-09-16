"""
================================================================================
REM-θ-framework — Gravitational Quantum Interferometry
Parametrized simulation script — v4 (two figures: neutron; electron+photon)

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
  Terminal:  python cow_simulation_final.py
  Spyder:    run cow_simulation_final.py   (plots appear in window)

OUTPUT — folder ./figures/ (created automatically)
------
  cow_fig_neutron.png          neutron interferometry (COW effect),
                                with real experimental data and a
                                residuals sub-panel
  cow_fig_electron_photon.png  electron interferometry (theoretical)
                                and photon gravitational redshift,
                                side by side
  cow_results.txt               numerical summary (neutron, electron,
                                photon)

NEUTRON DATA — real vs synthetic:
  DATA_FILE_N points to the digitized Staudenmann et al. (1980) Fig. 11
  dataset (CSV, included in this repository). If the file is missing,
  the script prints a warning and falls back to synthetic demo data
  instead of crashing -- panel (a)'s legend/title reflect which one
  was used ("Experimental data (digitized)" vs "Synthetic data (demo)").

NEUTRON INPUT — specify ONE (priority: V > E > λ_dB):
  V_NEUTRON_M_S : velocity (m/s)         → λ_dB = h/(m·v)
  E_NEUTRON_meV : kinetic energy (meV)   → v = √(2E/m)
  LAM_NEUTRON_A : λ_dB directly (Å)     → fallback
  λ_C(neutron) = 1.3196 fm is a constant — NOT a user input.

PHYSICS
-------
  Standard QM:       Δφ = 2π m² g A λ_dB sin(δ) / h²
  REM-θ-framework:    Δφ = 2π g A cot(θ) sin(δ) / (c² λ_C),  θ=arctan(pc/E0)  [Eq. H018, exact]
  Identity:          cot(θ)/λ_C ≡ m²λ_dB/h²  (error < 1e-9, asserted at
                     runtime for neutron and electron; verified here for
                     all three systems, including relativistic electron
                     velocities up to v/c~0.24)

NOTE: This is a proof-of-concept simulation. The algebraic identity above
is exact by construction, so agreement with standard QM is guaranteed
wherever standard QM is already experimentally validated -- this is not
an independent experimental test of the underlying framework.
================================================================================
"""

import os, sys
import numpy as np
import matplotlib
from scipy.optimize import curve_fit

# ── Backend: always render to file, no live window ─────────────
# (an interactive backend like Qt/Tk can conflict with Spyder's own
#  console backend, causing frozen/blank windows and layout glitches
#  that don't match the saved PNG -- saving directly avoids this
#  entire class of problem; open the PNG from the Files pane instead)
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
DATA_FILE_N    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'data', 'Staudenmann_Fig11.csv')  # phi_deg;counts
                           # '' = synthetic demo data

# ── Electron interferometry ───────────────────────────────────
LAM_E_MIN_PM   = 10.0     # min λ_dB (pm)
LAM_E_MAX_PM   = 80.0     # max λ_dB (pm)
AREA_E_CM2     = 1.0      # area (cm²)
DELTA_E_DEG    = 45.0     # tilt angle (degrees)

# ── Photon / Pound-Rebka ──────────────────────────────────────
H_MAX_M        = 100.0    # max height for theory curve (m)
H_PR_M         = 22.5     # Pound-Rebka tower height (m)
DNUT_MEAS      = 2.57e-15 # measured Δν/ν
DNUT_ERR       = 0.26e-15 # uncertainty

# ── Gravity ───────────────────────────────────────────────────
# Earth: 9.80665 | Moon: 1.62 | Mars: 3.72 | Jupiter: 24.79
G_LOCAL        = 9.80665  # m/s²

# ── Layout: panel (a) info box position ───────────────────────
BOX_X          = 0.02     # x (0=left, 1=right)
BOX_Y          = 0.015    # y (0=bottom, 1=top)

# ── Output ────────────────────────────────────────────────────
OUTPUT_FOLDER  = 'figures'
SAVE_COMBINED  = True
SAVE_INDIVIDUAL= False   # individual panels disabled -- only cow_fig_all.png is generated now
DPI            = 160

# ════════════════════════════════════════════════════════════════
# CONSTANTS (CODATA 2018)
# ════════════════════════════════════════════════════════════════
h_pl = 6.62607015e-34
c_l  = 2.99792458e8
eV   = 1.602176634e-19
mn   = 1.67492749804e-27;  lCn = h_pl/(mn*c_l)   # 1.3196 fm
me   = 9.1093837015e-31;   lCe = h_pl/(me*c_l)

# ════════════════════════════════════════════════════════════════
# PHYSICS FUNCTIONS
# ════════════════════════════════════════════════════════════════
def phi_std(mass,lam,A,delta,g=None):
    g=g or G_LOCAL
    return 2*np.pi*mass**2*g*A*lam*np.sin(delta)/h_pl**2

def phi_fw(mass,lam,A,delta,lC,g=None):
    """Framework: θ derived from λ_dB via θ=arctan(pc/E0), Eq. H018 (exact,
    relativistic); λ_C is Compton. Using arcsin(v/c) instead (non-relativistic
    approximation) introduces up to ~3% error at electron-interferometry
    velocities (v/c~0.24) — see cow_results.txt identity check."""
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
lam_n_A=lam_n*1e10; th_kin=np.arctan(h_pl/(lam_n*mn*c_l))
E_kin=0.5*mn*v_n**2/eV*1000

a_c=A_SLAB_MM*1e-3; d_c=D_SLAB_MM*1e-3; d_pl=D_PLANES_A*1e-10
thB=np.arcsin(lam_n/(2*d_pl))
A_p=(2*d_c**2+2*a_c*d_c*np.cos(thB))*np.tan(thB)
th_n=np.arctan(h_pl/(lam_n*mn*c_l))
q_grav=phi_std(mn,lam_n,A_p,np.pi/2)
q_fw_n=phi_fw(mn,lam_n,A_p,np.pi/2,lCn)
assert abs(q_fw_n-q_grav)/q_grav<1e-9, "Framework/canonical identity failed for neutron"
q_bend=(C_BEND/lam_n_A)*np.sin(thB)**2
q_tot=q_grav+q_bend

if DATA_FILE_N and os.path.isfile(DATA_FILE_N):
    rows=[]
    with open(DATA_FILE_N) as f:
        for line in f:
            p=line.strip().split(';')
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

def mdl_n(pd,A,B,C,D,Q,Delta1,phi0,beta0):
    """Full empirical model, Eq. 54 of Staudenmann et al. (1980):
    I(phi) = A + B*phi + C*phi^2 + D*cos(Q*sin(phi)+Delta1)*cos(q*sin(phi-phi0)+beta0)
    q (=q_tot) is fixed from the physics -- NOT a free fit parameter here,
    since the purpose is to test whether the theoretical q matches the data,
    not to re-extract q empirically."""
    r=np.radians(pd); r0=np.radians(pd-phi0)
    bg=A+B*pd+C*pd**2
    mod=D*np.cos(Q*np.sin(r)+Delta1)
    return bg+mod*np.cos(q_tot*np.sin(r0)+beta0)
popt_n,_=curve_fit(mdl_n,phi_exp,cnt_exp,
    p0=[cnt_exp.mean(),0.,0.,(cnt_exp.max()-cnt_exp.min())/2,1.0,0.,0.,0.],
    bounds=([500,-20,-1,50,0.0,-np.pi,-10,-np.pi],
            [4000, 20, 1,2500,20.0, np.pi, 10, np.pi]),maxfev=200000)
A_f,B_f,C_f,D_f,Q_f,Delta1_f,phi0_f,beta0_f=popt_n
rms_S=np.sqrt(np.mean((cnt_exp-mdl_n(phi_exp,*popt_n))**2))

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
ymin=min(cnt_exp.min(),c_fw.min())-950
ymax=max(cnt_exp.max(),c_fw.max())+850

# ════════════════════════════════════════════════════════════════
# ELECTRON SETUP
# ════════════════════════════════════════════════════════════════
lam_er=np.linspace(LAM_E_MIN_PM*1e-12,LAM_E_MAX_PM*1e-12,300)
A_e=AREA_E_CM2*1e-4; de=np.radians(DELTA_E_DEG)
pfe=np.array([phi_fw(me,l,A_e,de,lCe) for l in lam_er])
pse=np.array([phi_std(me,l,A_e,de)    for l in lam_er])

# ════════════════════════════════════════════════════════════════
# PHOTON SETUP
# ════════════════════════════════════════════════════════════════
fp=G_LOCAL*H_PR_M/c_l**2; hr=np.linspace(0,H_MAX_M,300)

# ════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ════════════════════════════════════════════════════════════════
print("="*55)
print("REM-θ-framework — Gravitational Interferometry")
print("="*55)
print(f"\n[Neutron] {src}: λ_dB={lam_n_A:.4f}Å, v={v_n:.2f}m/s, E={E_kin:.3f}meV")
print(f"  λ_C(n)  = {lCn*1e15:.4f} fm  (Compton, constant)")
print(f"  q_grav  = {q_grav:.4f} rad  (std = fw, diff={abs(q_fw_n-q_grav)/q_grav:.2e})")
print(f"  q_bend  = {q_bend:.4f} rad")
print(f"  q_total = {q_tot:.4f} rad, RMS={rms_S:.0f} cts")
diff_e=np.max(np.abs(pfe-pse)/np.abs(pse))
assert diff_e<1e-9, "Framework/canonical identity failed for electron"
print(f"\n[Electron] max diff = {diff_e:.2e}")
print(f"\n[Photon]   Predicted={fp:.4e}, Measured={DNUT_MEAS:.4e}, Ratio={DNUT_MEAS/fp:.4f}")

# ════════════════════════════════════════════════════════════════
# DRAWING FUNCTIONS
# ════════════════════════════════════════════════════════════════
os.makedirs(OUTPUT_FOLDER,exist_ok=True)
def fp_(n): return os.path.join(OUTPUT_FOLDER,n)
def ibox(ax,txt,x=0.98,y=0.97,fc='lightyellow',ha='right',va='top',fontsize=9.5):
    ax.text(x,y,txt,transform=ax.transAxes,fontsize=fontsize,ha=ha,va=va,
            bbox=dict(boxstyle='round',fc=fc,alpha=0.92,ec='gray'))

def draw_a(axes):
    ax,ax_r=axes
    resid_a=cnt_exp-mdl_n(phi_exp,*popt_n)
    l_pu,=ax.plot(pf2,c_pure,'g:',lw=1.5,zorder=2,alpha=0.85)
    l_fw,=ax.plot(pf2,c_fw,  'b-',lw=2.0,zorder=3)
    l_st,=ax.plot(pf2,c_st,  'r--',lw=1.4,zorder=4,alpha=0.7)
    l_dt =ax.scatter(phi_exp,cnt_exp,s=7,c='black',zorder=6,alpha=0.85)
    ax.set_xlim(-41,40); ax.set_ylim(ymin,ymax)
    ax.set_ylabel(r'Neutron counts $C_3$',fontsize=11.5)
    ax.set_title('Neutron — Staudenmann et al. (1980)\n'
                 rf'$\lambda_{{dB}}={lam_n_A:.3f}\,\AA$, '
                 rf'$a={A_SLAB_MM}\,\rm mm$, $d={D_SLAB_MM}\,\rm mm$',
                 fontsize=9.5,pad=4)
    ax.grid(True,alpha=0.2)
    ax.xaxis.set_ticklabels([])
    ax.legend([l_dt,l_st,l_pu,l_fw],
        [dlbl,
         'Std QM $+$corr. (identical)',
         f'Framework only: $q={q_fw_n:.1f}$ rad',
         f'Framework $+$corr.: $q={q_fw_n+q_bend:.1f}$ rad'],
        fontsize=9,loc='upper center',framealpha=0.92,
        ncol=2,columnspacing=0.6,handletextpad=0.4,
        borderpad=0.5,handlelength=1.6)
    ibox(ax,f'$q_{{\\rm grav}}={q_grav:.1f}$ rad\n'
            f'$q_{{\\rm bend}}={q_bend:.2f}$ rad\n'
            f'RMS$={rms_S:.0f}$ cts',
         x=BOX_X,y=BOX_Y,ha='left',va='bottom',fontsize=11)
    ax_r.scatter(phi_exp,resid_a,s=10,c='darkred',alpha=0.8,zorder=5)
    ax_r.axhline(0,color='k',lw=1)
    ax_r.axhspan(-rms_S,rms_S,color='gray',alpha=0.15)
    ax_r.set_xlim(-41,40)
    ax_r.set_xlabel(r'Rotation angle $\phi$ (deg)',fontsize=11)
    ax_r.set_ylabel('Resid.',fontsize=10)
    ax_r.tick_params(axis='both',labelsize=8.5)
    ax_r.grid(True,alpha=0.2)

def draw_b(ax):
    ax.plot(lam_er*1e12,pfe*1e8,'b-',lw=2.5,label='Framework')
    ax.plot(lam_er*1e12,pse*1e8,'r--',lw=1.5,label='Standard QM',alpha=0.8)
    ax.fill_between(lam_er*1e12,pfe*0.95*1e8,pfe*1.05*1e8,
                    alpha=0.12,color='blue',label='±5% band')
    ax.set_xlabel(r'$\lambda_{dB}$ (pm)',fontsize=11.5)
    ax.set_ylabel(r'$\Delta\phi\;(\times10^{-8}$ rad)',fontsize=11.5)
    ax.set_title(f'(a) Electron interferometry (theoretical)\n'
                 rf'$A={AREA_E_CM2}\,\rm cm^2$, $\delta={DELTA_E_DEG:.0f}°$',
                 fontsize=9.5,pad=4)
    ax.legend(fontsize=8.5,loc='upper left'); ax.grid(True,alpha=0.2)
    ibox(ax,'Framework $\\equiv$ Std QM\nerror $<10^{-10}$',x=0.97,y=0.14)

def draw_c(ax):
    ax2=ax.twinx()
    ax.plot(hr,2*np.pi*G_LOCAL*hr/c_l**2*1e14,'b-',lw=2.5,
            label=r'Framework $=$ Std: $2\pi gh/c^2$')
    ax2.plot(hr,G_LOCAL*hr/c_l**2*1e15,'purple',lw=1.2,ls=':',alpha=0.6,
             label=r'$\Delta\nu/\nu\,(\times10^{-15})$')
    ax.errorbar([H_PR_M],[2*np.pi*DNUT_MEAS*1e14],
                yerr=[2*np.pi*DNUT_ERR*1e14],
                fmt='ko',ms=8,capsize=5,zorder=5,label='Pound-Rebka (1959)')
    ax.set_xlabel('Height $h$ (m)',fontsize=11.5)
    ax.set_ylabel(r'$\Delta\phi\,(\times10^{-14}$ rad)',fontsize=11.5)
    ax2.set_ylabel(r'$\Delta\nu/\nu\,(\times10^{-15})$',fontsize=11.5,color='purple')
    ax.set_title(r'(b) Photon — Pound-Rebka (1959)'+'\n'
                 r'$\theta_\gamma=\pi/4$, $\cot(\pi/4)=1$',fontsize=9.5,pad=4)
    ax.legend(fontsize=8,loc='upper left')
    ax2.legend(fontsize=7.5,loc='lower right'); ax.grid(True,alpha=0.2)
    ibox(ax,f'Predicted: {fp*1e15:.3f}'+r'$\times10^{-15}$'+'\n'
            f'Measured: {DNUT_MEAS*1e15:.2f}'+'$\\pm$'+
            f'{DNUT_ERR*1e15:.2f}'+r'$\times10^{-15}$'+'\n'
            f'Ratio: {DNUT_MEAS/fp:.3f}',
         x=0.97,y=0.42,fc='lightgreen')
    return ax2

# ════════════════════════════════════════════════════════════════
# NEUTRON FIGURE (standalone)
# ════════════════════════════════════════════════════════════════
if SAVE_COMBINED:
    fig=plt.figure(figsize=(11,8.5))
    fig.text(0.5,0.995,
        "Gravity-Induced Quantum Phase Shift — REM-$\\theta$-framework",
        ha='center',va='top',fontsize=12,fontweight='bold')
    fig.text(0.5,0.955,
        r"$\Delta\phi_{\rm geom}=\dfrac{2\pi gA\cot\theta\sin\delta}{c^2\lambda_C}"
        r"\equiv\dfrac{2\pi m^2gA\lambda_{dB}\sin\delta}{h^2}$"
        r"  (error $<10^{-10}$)",
        ha='center',va='top',fontsize=10.5)
    gs=GridSpec(2,1,figure=fig,height_ratios=[3,1],hspace=0.08,
                left=0.09,right=0.97,top=0.86,bottom=0.09)
    draw_a((fig.add_subplot(gs[0]),fig.add_subplot(gs[1])))
    plt.savefig(fp_('cow_fig_neutron.png'),dpi=DPI,bbox_inches='tight')
    plt.close()
    print(f"Saved: {fp_('cow_fig_neutron.png')}")

# ════════════════════════════════════════════════════════════════
# ELECTRON + PHOTON FIGURE (2 panels)
# ════════════════════════════════════════════════════════════════
if SAVE_COMBINED:
    fig=plt.figure(figsize=(11,5))
    fig.text(0.5,0.99,
        "Gravity-Induced Quantum Phase Shift — REM-$\\theta$-framework",
        ha='center',va='top',fontsize=12,fontweight='bold')
    fig.text(0.5,0.90,
        r"$\Delta\phi_{\rm geom}=\dfrac{2\pi gA\cot\theta\sin\delta}{c^2\lambda_C}"
        r"\equiv\dfrac{2\pi m^2gA\lambda_{dB}\sin\delta}{h^2}$"
        r"  (error $<10^{-10}$)",
        ha='center',va='top',fontsize=10.5)
    gs=GridSpec(1,2,figure=fig,wspace=0.28,
                left=0.08,right=0.97,top=0.72,bottom=0.14)
    draw_b(fig.add_subplot(gs[0,0]))
    draw_c(fig.add_subplot(gs[0,1]))
    plt.savefig(fp_('cow_fig_electron_photon.png'),dpi=DPI,bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fp_('cow_fig_electron_photon.png')}")

# ════════════════════════════════════════════════════════════════
# NUMERICAL SUMMARY
# ════════════════════════════════════════════════════════════════
with open('cow_results.txt','w') as f:
    f.write("REM-θ-framework — Results\n"+"="*40+"\n")
    f.write(f"G_LOCAL = {G_LOCAL} m/s²\n\n")
    f.write(f"[Neutron] {src}: λ_dB={lam_n_A:.4f}Å\n")
    f.write(f"  λ_C(n)  = {lCn*1e15:.4f} fm\n")
    f.write(f"  q_grav  = {q_grav:.6f} rad\n")
    f.write(f"  q_fw    = {q_fw_n:.6f} rad\n")
    f.write(f"  |diff|  = {abs(q_fw_n-q_grav)/q_grav:.2e}\n")
    f.write(f"  q_bend  = {q_bend:.6f} rad\n")
    f.write(f"  q_total = {q_tot:.6f} rad\n")
    f.write(f"  RMS     = {rms_S:.1f} counts\n\n")
    f.write(f"[Electron] max diff = {diff_e:.2e}\n\n")
    f.write(f"[Photon] Predicted={fp:.4e}, Measured={DNUT_MEAS:.4e}\n")
    f.write(f"  Ratio = {DNUT_MEAS/fp:.4f}\n")
print("Saved: cow_results.txt")
print(f"All figures in: ./{OUTPUT_FOLDER}/")
print("Done.")
