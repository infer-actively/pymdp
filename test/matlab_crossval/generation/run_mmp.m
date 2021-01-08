function [F, G, x, xq, vn, xn] = run_mmp(num_iter, window_len, policy_matrix, t, xq, x, L, D, b, b_t, xn, vn)
%run_mmp Functioned out version of the marginal message passing routine
%that happens in Karl's SPM_MDP_VB_X.m

% marginal message passing (MMP)
%--------------------------------------------------------------

num_factors = length(xq);
num_states = zeros(1,num_factors);
for f = 1:num_factors
    num_states(f) = size(xq{f},1);
end

S     = size(policy_matrix,1) + 1;   % horizon
R = t;

dF    = 1;                  % reset criterion for this policy
for iter = 1:num_iter       % iterate belief updates
    F  = 0;                 % reset free energy for this policy
    for j = max(1,t-window_len):S             % loop over future time points
        
        % curent posterior over outcome factors
        %--------------------------------------------------
        if j <= t
            for f = 1:num_factors
                xq{f} = x{f}(:,j,1);
            end
        end
        
        for f = 1:num_factors
            
            % hidden states for this time and policy
            %----------------------------------------------
            sx = x{f}(:,j,1);
            qL = zeros(num_states(f),1);
            v  = zeros(num_states(f),1);
            
            % evaluate free energy and gradients (v = dFdx)
            %----------------------------------------------
            if dF > exp(-8) || iter > 4
                
                % marginal likelihood over outcome factors
                %------------------------------------------
                if j <= t
                    qL = spm_dot(L{j},xq,f);
                    qL = spm_log(qL(:));
                end                              
                
                % entropy
                %------------------------------------------
                qx  = spm_log(sx);
                
                % emprical priors (forward messages)
                %------------------------------------------
                if j < 2
                    px = spm_log(D{f});
                    v  = v + px + qL - qx;
                else
                    px = spm_log(b{f}(:,:,policy_matrix(j - 1,1,f))*x{f}(:,j - 1,1));
                    v  = v + px + qL - qx;
                end               
              
                
                % emprical priors (backward messages)
                %------------------------------------------
                if j < R
                    px = log( b_t{f}(:,:,policy_matrix(j,1,f)) * x{f}(:,j+1,1) );
%                     if iter == num_iter
%                         fprintf('inference timestep: %d, factor: %d \n',j, f)
%                         disp(px)
%                     end
                    v  = v + px + qL - qx;
                end
                
                % (negative) free energy
                %------------------------------------------
                if j == 1 || j == S
                    F = F + sx'*0.5*v;
                else
                    F = F  + sx'*(0.5*v - (num_factors-1)*qL/num_factors);
                end
                
                % update
                %-----------------------------------------                
                
                v    = v - mean(v);
%                 if iter == num_iter
%                     fprintf('inference timestep: %d, factor: %d \n',j, f)
%                     disp(v)
%                 end
                
                sx   = softmax(qx + v/4);
                
            else
                F = G;
            end
            
            % store update neuronal activity
            %----------------------------------------------
            x{f}(:,j,1)          = sx;
            xq{f}                = sx;
            xn{f}(iter,:,j,t,1)  = sx;
            vn{f}(iter,:,j,t,1)  = v;
            
        end
    end
    
    % convergence
    %------------------------------------------------------
    if iter > 1
        dF = F - G;
    end
    G = F;
    
end

end

function A  = spm_log(A)
% log of numeric array plus a small constant
%--------------------------------------------------------------------------
A  = log(A + 1e-16);
end

function A  = spm_norm(A)
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A           = bsxfun(@rdivide,A,sum(A,1));
A(isnan(A)) = 1/size(A,1);
end

function [X] = spm_dot(X,x,i)
% Multidimensional dot (inner) product

% initialise dimensions
%--------------------------------------------------------------------------
if iscell(x)
    DIM = (1:numel(x)) + ndims(X) - numel(x);
else
    DIM = 1;
    x   = {x};
end

% omit dimensions specified
%--------------------------------------------------------------------------
if nargin > 2
    DIM(i) = [];
    x(i)   = [];
end

% inner product using recursive summation (and bsxfun)
%--------------------------------------------------------------------------
for d = 1:numel(x)
    s         = ones(1,ndims(X));
    s(DIM(d)) = numel(x{d});
    X         = bsxfun(@times,X,reshape(full(x{d}),s));
    X         = sum(X,DIM(d));
end

% eliminate singleton dimensions
%--------------------------------------------------------------------------
X = squeeze(X);
end

function [y] = spm_softmax(x,k)
% softmax (e.g., neural transfer) function over columns

% apply
%--------------------------------------------------------------------------
if nargin > 1,    x = k*x; end
if size(x,1) < 2; y = ones(size(x)); return, end

% exponentiate and normalise
%--------------------------------------------------------------------------
x  = exp(bsxfun(@minus,x,max(x)));
y  = bsxfun(@rdivide,x,sum(x));
end

