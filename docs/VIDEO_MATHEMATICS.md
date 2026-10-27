# The mathematics explained in the demonstration

The film connects a familiar question—“has the way these parts move changed?”—to several different mathematical checks. The animated examples explain the method. They are not additional measurements from the short fan recordings.

## Watch the explanation

Approximate times in the combined film:

| Time | Idea | Practical contribution |
|---|---|---|
| 1:25 | Welch power spectrum | Separates repeating motion into frequencies, within the sampling limit |
| 1:55 | Joint spectral matrix | Preserves whether visible parts move together or in opposition |
| 2:14 | Mechanical model | Relates motion to mass, damping and reciprocal spring couplings |
| 2:31 | Scale-invariant operator | Cancels fixed positive image scales under the stated model |
| 2:53 | Eigenvalues and eigenvectors | Connects the recovered operator to natural frequencies and mode shapes |
| 3:12 | Estimated uncertainty | Supports collecting more evidence, resolving a change or abstaining |

## Equal strength can hide different coordination

Two equal-amplitude sinusoidal channels can move together or in opposition. Their individual power spectra are identical. Their normalized joint spectral matrices at that frequency are

$$S_+=\begin{pmatrix}1&1\\1&1\end{pmatrix},\qquad S_-=\begin{pmatrix}1&-1\\-1&1\end{pmatrix}.$$

Both have eigenvalues 0 and 2, but their leading eigenvectors differ: $(1,1)/\sqrt2$ versus $(1,-1)/\sqrt2$. The joint relationship contains information that the two separate power spectra discard. These are operational motion patterns; they are not automatically identified natural modes or proof of damage.

## A close-up changes pixels, not the machine

The mechanical pathway assumes

$$M\ddot x+C\dot x+Kx=f,\qquad y=Sx+\eta,$$

with positive diagonal mass $M$, fixed positive diagonal image scales $S$, reciprocal spring couplings $K_{ij}\leq0$, and sufficient observable coordinates and modes. Camera compensation and identification quality remain separate requirements.

The explanatory example sets

$$M=I,\quad K=\begin{pmatrix}6&-2\\-2&4\end{pmatrix},\quad S=\operatorname{diag}(2,1).$$

The pixel-coordinate stiffness operator becomes

$$B_y=SM^{-1}KS^{-1}=\begin{pmatrix}6&-4\\-1&4\end{pmatrix}.$$

The reciprocal coefficients change from $-2,-2$ to $-4,-1$. Their product stays 4. In the diagram, an arrow from B to A represents the row-A/column-B coefficient. Defining

$$D_{ii}=(B_y)_{ii},\qquad D_{ij}=-\sqrt{(B_y)_{ij}(B_y)_{ji}}\quad(i\ne j)$$

recovers $D=M^{-1/2}KM^{-1/2}=K$ in this example. The result is invariant to these fixed positive region scales. This is not invariance to arbitrary camera transformations, moving scale factors or out-of-plane motion, and it does not recover calibrated masses, forces or physical stiffness separately.

## The operator has its own spectrum

For $D\varphi_k=\omega_k^2\varphi_k$, the eigenvalues describe squared natural angular frequencies of the compatible conservative mechanical model; the eigenvectors describe mass-normalized mode shapes. This operator spectrum is distinct from a measured signal's power spectrum.

The illustrated grounded system has baseline eigenvalues approximately 2.76 and 7.24 in normalized model units. Reducing its inter-part spring from 2 to 1.2 changes $K$ to

$$K_{\rm weak}=\begin{pmatrix}5.2&-1.2\\-1.2&3.2\end{pmatrix},$$

whose eigenvalues are approximately 2.64 and 5.76. Ground springs remain 4 and 2. The animation uses the eigenvectors of the intermediate model as the joint is weakened. Displacements are exaggerated for visibility; these are not measured fan frequencies.

## Resolve a change before attempting localization

The model example has operator change $\|K_{\rm weak}-K\|_2=1.6$. With combined illustrative uncertainty radius 1.8 it is unresolved; with radius 0.2 it is resolved. The film animates selected radii to explain this comparison. They are not fitted bootstrap values or an experiment showing that more footage always reduces uncertainty.

The application's multichannel block-bootstrap radii are empirical estimates. They do not certify a coverage probability, and they exclude model bias and nonstationarity. A resolved operator change alone does not prove a location or cause: physical compatibility, mode separation, entry support and excitation assumptions still matter. Unknown driving forces can prevent causal structural identification.

## What was checked

The film source script `prepare_math.py` (in the separate story2 production folder) verifies the scale identity across 100 randomized positive rescalings, the equal-power phase counterexample, the eigenpairs, and both uncertainty decisions against the application's `invariant_operator` and `compare_operators` functions. Numeric values and assumptions are exported as [math-illustration.json](../media/math-illustration.json). Movie encoding, full decoding, timing and source hashes are checked separately.

The real fan result remains a 35.9× tone-amplitude ratio in the recorded AWS job. Those short clips observe one housing and do not reconstruct the mechanical operator or identify a damaged bearing. Their structural verdict remains abstention.

See the [technical report](TECHNICAL_REPORT.md) and [full mathematical appendix](MATHEMATICS.md) for the implementation conditions and derivations.
