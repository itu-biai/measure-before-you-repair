"""What the model reads, word for word (D137): the system prompt, the first message, the router's prompt and the
twelve tool cards, as approved in the Task 4 Stage 1 before-doc (visual-guide/task4/01_harness/1_before.html §3, §5,
§8), then changed in Stage 2 one change at a time (D145; each change and its reason in
visual-guide/task4/02_develop/changes.md). The numbers in the cards were counted there from the Task 2 findings and Task 3 scores; they are copied here so
nothing is read at run time, and a test checks them against before_numbers.json. The router's placeholder "[same
protocol lines as the agent]" is the system prompt's protocol paragraph."""

RESCAN = 0.033  # motion-tool value at which half of MR-ART's scans were graded unusable (D123, narai2022)

SYSTEM = (
    'You restore one 3-D T1-weighted brain MRI volume. It may contain any mix of five artifacts, or none: noise, RF '
    'spikes (stripes), Nyquist ghosting (copies along the phase-encoding direction), bias field (smooth shading) and '
    'head motion. You cannot see a clean scan of this head.\n'
    '\n'
    'You have tools to look at and compare versions, five tools that each measure one artifact, and four that each '
    'repair one (noise, spikes, ghosting, bias). Every repair makes a new version (v1, v2, …); earlier versions '
    "stay, and you may go on from any of them. Each tool's description says what it measures and how it did on "
    "development volumes. A repair's report gives its tool's reading afterwards: an artifact can remain after one "
    'repair, and the readings of the others can change once it is gone.\n'
    '\n'
    'There is no repair for motion. When motion would make the scan unusable for diagnosis, say a rescan is needed: '
    'a motion-tool value of 0.033 or more is the level at which radiologists, in the MR-ART study, judged half of '
    'such scans unusable.\n'
    '\n'
    'Scanner protocol: readout S–I; phase encoding A–P (slow) and L–R (fast); 1 mm voxels; intensities scaled so '
    'white matter = 1; axes RAS.\n'
    '\n'
    'When you are done, call finish with the version you return and your report.')
FIRST = (
    'Here is the volume as received, v0: axial, coronal and sagittal slices through the middle of the head.')
PROTOCOL = (
    'Scanner protocol: readout S–I; phase encoding A–P (slow) and L–R (fast); 1 mm voxels; intensities scaled so '
    'white matter = 1; axes RAS.')
ROUTER = (
    'You see one 3-D T1-weighted brain MRI volume (three slices below). It may contain any mix of five artifacts, or '
    'none: noise, RF spikes (stripes), Nyquist ghosting, bias field (smooth shading), head motion. Write what you '
    'see, then answer yes or no for each artifact, and whether motion makes a rescan necessary. Scanner protocol: '
    'readout S–I; phase encoding A–P (slow) and L–R (fast); 1 mm voxels; intensities scaled so white matter = 1; '
    'axes RAS.')
ROUTER_READINGS = (  # ③b (D148): the router's words, with the readings and the rescan level the agent is given
    'You see one 3-D T1-weighted brain MRI volume (three slices below), then the readings of five tools on it, each '
    'measuring one artifact, with what the tool does and its picture. It may contain any mix of five artifacts, or '
    'none: noise, RF spikes (stripes), Nyquist ghosting, bias field (smooth shading), head motion. Write what you see '
    'and what the readings show, then answer yes or no for each artifact, and whether motion makes a rescan '
    'necessary: a motion-tool value of 0.033 or more is the level at which radiologists, in the MR-ART study, judged '
    'half of such scans unusable. ' + PROTOCOL)
NUDGE = "Call one of the functions; finish ends the run."  # the harness's only line of its own (a reply with no call)

CARDS = {
    'look': (
        'Shows a version: axial, coronal and sagittal slices through the middle of the head, grey scale 0 to 1.35 × '
        'white matter. Optional: one plane and a slice number, and the air window (0 to 0.1 × white matter), which '
        'shows faint signal outside the head: noise, copies of the head, ripples from motion.'),
    'compare': (
        'Shows version a minus version b on the same three slices (red: a brighter, blue: b brighter) and gives the '
        'RMS change inside the head (white matter = 1).'),
    'noise_tool': (
        'Measures the noise level σ, as a fraction of white matter, from many small patches of flat tissue. Says yes '
        'at σ ≥ 0.030. On 60 dev volumes with mixed artifacts: yes on 29 of 29 with added noise, and on 3 of 31 '
        'without (scans whose own noise is high). Returns yes/no, the value, a confidence, the reason, and a picture '
        'of where it measured.'),
    'spike_tool': (
        "Looks for isolated bright points in k-space (the scan's raw frequency data); each one makes a stripe "
        'pattern across the image. Says yes at z ≥ 7.0. On 60 dev volumes with mixed artifacts: yes on 28 of 29 with '
        'spikes (a faint one missed), on 0 of 31 without.'),
    'ghosting_tool': (
        'Looks for a repeating rhythm along A–P (the phase-encoding direction) and faint copies of the head in the '
        'air. Says yes when either is found. On 60 dev volumes with mixed artifacts: yes on 28 of 29 with ghosting, '
        'on 0 of 31 without.'),
    'bias_tool': (
        "Fits a smooth shading field over the brain (N4). The value is the shading's max/min − 1; says yes at ≥ "
        '0.083. On 60 dev volumes with mixed artifacts: yes on 28 of 29 with shading, on 0 of 31 without.'),
    'motion_tool': (
        'Measures head signal leaking into the air along A–P; the value is the whole-brain damage as a fraction of '
        'white matter. Says yes at ≥ 0.0038. The rescan level is 0.033 (see the task). On 60 dev volumes with mixed '
        'artifacts, on the volume as received: of 29 with motion, yes on 6, no on 10, cannot tell on 13 (it cannot '
        'tell while ghosting is present); after the other repairs it could not tell on 1 of 29. It under-reads '
        'motion when other artifacts are present: of the dev volumes whose motion is at or above the rescan level, '
        'it read that high on 1 of 5 with other artifacts (after the other repairs) and on 8 of 11 with motion '
        'alone. Once a spike repair raised its reading from 0.002 to 0.080.'),
    'fix_spike': (
        'Removes the stripes of each spike the spike tool finds, and returns a new version with a report (what '
        "changed, and the spike tool's reading after). If the spike tool says no, nothing changes. On 32 dev volumes "
        'with spikes alone: median 98 % of the spike error removed, none made worse, about 19 s.'),
    'fix_ghosting': (
        'Puts back the signal that ghosting moved out of the head along A–P, and returns a new version with a '
        'report. If the ghosting tool says no, nothing changes. On 32 dev volumes with ghosting alone: median 97 % '
        'removed, none made worse, about 14 s.'),
    'fix_noise': (
        'Denoises with a pretrained network (DRUNet) at the σ the noise tool measured, and returns a new version '
        'with a report. If the noise tool says no, nothing changes. On 32 dev volumes with noise alone: median 42 % '
        "of the added noise removed (the scan's own noise is not counted as error), none made worse, about 40 s. It "
        'also changes the 16 of 210 dev volumes where the noise tool says yes without added noise.'),
    'fix_bias': (
        'Divides out the smooth shading the bias tool fits (N4), and returns a new version with a report. If the '
        'bias tool says no, nothing changes. On 32 dev volumes with shading alone: median 71 % removed, none made '
        'worse, about 12 s.'),
    'finish': (
        'Ends the run: your report, written as a radiologist writes one, and the version you return.'),
}

# ④a, the closed loop without probes (D150): ④'s texts with the measuring tools taken out. Paragraphs 1 and 2 of SYSTEM
# are the only ones edited (a test holds the rest equal); the repair cards keep their development numbers and say what
# a repair returns when it finds nothing to repair. Re-derived from ④'s final texts at the freeze (r02's sentence without
# the readings: "An artifact can remain after one repair.").
EYES_EDITED = [1, 2]
_SYS = SYSTEM.split('\n\n')
SYSTEM_EYES = '\n\n'.join([
    _SYS[0],
    'You have tools to look at and compare versions, and four that each repair one artifact (noise, spikes, ghosting, '
    'bias). Every repair makes a new version (v1, v2, …); earlier versions stay, and you may go on from any of them. '
    "Each tool's description says what it does and how it did on development volumes. An artifact can remain after "
    'one repair.',
    'There is no repair for motion. When motion would make the scan unusable for diagnosis, say a rescan is needed.',
    *_SYS[3:]])
CARDS_EYES = {
    'look': CARDS['look'],
    'compare': CARDS['compare'],
    'fix_spike': (
        'Removes the stripes of each RF spike it finds in k-space and returns a new version, or the same version '
        'unchanged when it finds none.' + ' On 32 dev volumes' + CARDS['fix_spike'].split('On 32 dev volumes')[1]),
    'fix_ghosting': (
        'Puts back the signal that ghosting moved out of the head along A–P and returns a new version, or the same '
        'version unchanged when it finds no ghosting.' + ' On 32 dev volumes'
        + CARDS['fix_ghosting'].split('On 32 dev volumes')[1]),
    'fix_noise': (
        'Denoises with a pretrained network (DRUNet) at the noise level it measures and returns a new version, or the '
        'same version unchanged when the noise is low.' + ' On 32 dev volumes'
        + CARDS['fix_noise'].split('On 32 dev volumes')[1].replace(
            'where the noise tool says yes without added noise', 'whose own noise is high')),
    'fix_bias': (
        'Divides out the smooth shading it fits over the brain (N4) and returns a new version, or the same version '
        'unchanged when the shading is no more than a clean brain has.' + ' On 32 dev volumes'
        + CARDS['fix_bias'].split('On 32 dev volumes')[1]),
    'finish': CARDS['finish'],
}
