%% a simple method for croping background region
function context_m = context_mask(pixels,target_sz)

    ratio = 0.75;
    w = size(pixels, 2);
    h = size(pixels, 1);
    hs = round(0.5*h-0.5*ratio*target_sz(1)):round(0.5*h+0.5*ratio*target_sz(1));
    ws = round(0.5*w-0.5*ratio*target_sz(2)):round(0.5*w+0.5*ratio*target_sz(2));
    mask = ones(w, h);
    mask(hs, ws) = 0;
    context_m = mask;
end