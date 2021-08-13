function [g_f] = run_training(model_xf,xcf,use_sz,params,yf,small_filter_sz)

    g_f = single(zeros(size(model_xf)));
    h_f = g_f;
    l_f = g_f;
    mu    = 1;
    betha = 10;
    mumax = 10000;
    i = 1;
    
    T = prod(use_sz);
    S_xx0 = sum(conj(model_xf) .* model_xf, 3);
    S_xx = S_xx0 + sqrt(params.yta) * sum(conj(xcf) .* xcf, 3);
    u_f = model_xf + sqrt(params.yta) * xcf;
    S_uu = sum(conj(u_f) .* model_xf, 3);
    
    %   ADMM
    while (i <= params.admm_iterations)
        %   solve for G- please refer to the paper for more details
        B = S_xx + (T * mu);
        S_lx = sum(conj(u_f) .* l_f, 3);
        S_hx = sum(conj(u_f) .* h_f, 3);
        g_f = (((1/(T*mu)) * bsxfun(@times, yf, model_xf)) - ((1/mu) * l_f) + h_f) - ...
            bsxfun(@rdivide,(((1/(T*mu)) * bsxfun(@times, u_f, (S_uu .* yf))) - ((1/mu) * bsxfun(@times, u_f, S_lx)) + (bsxfun(@times, u_f, S_hx))), B);
%         %%% ²»ÓÃsM
%         g_f = (model_xf .* conj(yf) + mu * T * h_f - T * l_f) ./(S_xx0 + params.yta * sum(conj(xcf) .* xcf, 3) + mu * T);
        
        %   solve for H
        h = (T/((mu*T)+ params.admm_lambda))* ifft2((mu*g_f) + l_f);
        [sx,sy,h] = get_subwindow_no_window(h, floor(use_sz/2) , small_filter_sz);
        t = single(zeros(use_sz(1), use_sz(2), size(h,3)));
        t(sx,sy,:) = h;
        h_f = fft2(t);
        
        %   update L
        l_f = l_f + (mu * (g_f - h_f));
        
        %   update mu- betha = 10.
        mu = min(betha * mu, mumax);
        i = i+1;
    end
    
end

