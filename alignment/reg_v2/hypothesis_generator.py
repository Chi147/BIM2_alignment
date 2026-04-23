import numpy as np

def get_base_scale(ifc_segs, pdf_segs):
    """
    Compares the average length of the 'main' walls in both files
    to estimate how much the IFC needs to be scaled up.
    """
    def get_main_walls(segs, top_n=15):
        if not segs: return 1.0
        # Sort by length and take the top 15 longest lines
        lengths = [np.sqrt((x2-x1)**2 + (y2-y1)**2) for x1,y1,x2,y2 in segs]
        # Ensure we don't try to grab more lines than we have
        actual_n = min(len(lengths), top_n)
        sorted_indices = np.argsort(lengths)[-actual_n:]
        return np.mean([lengths[i] for i in sorted_indices])
    
    ifc_avg = get_main_walls(ifc_segs)
    pdf_avg = get_main_walls(pdf_segs)
    
    return pdf_avg / ifc_avg

def generate_hypotheses(ifc_segs, pdf_segs, ifc_meta, pdf_meta):
    """
    Generates a grid of starting guesses for the solver.
    """
    # AUTOMATION FIX: Use the calculated scale instead of 58.0
    auto_scale = get_base_scale(ifc_segs, pdf_segs)
    
    pw, ph = pdf_meta.bbox[2], pdf_meta.bbox[3]
    hypotheses = []
    
    # 3x3 Grid across the PDF page
    xs = [pw * 0.25, pw * 0.5, pw * 0.75]
    ys = [ph * 0.25, ph * 0.5, ph * 0.75]
    
    # We test 3 variations of the auto-scale to be safe (0.9x, 1.0x, 1.1x)
    for s_mult in [0.9, 1.0, 1.1]:
        test_scale = auto_scale * s_mult
        
        for tx in xs:
            for ty in ys:
                # Test all 4 cardinal rotations
                for angle in [0, 90, 180, 270]:
                    hypotheses.append({
                        'scale': test_scale, 
                        'rotation': angle, 
                        'tx': tx, 
                        'ty': ty
                    })
    return hypotheses