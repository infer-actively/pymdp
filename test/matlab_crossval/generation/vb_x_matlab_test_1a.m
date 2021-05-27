%%% PSEUDO-CODE VERSION OF SPM_MDP_VB_X.m

clear all; close all; cd .. % this brings you into the 'pymdp/tests/matlab_crossval/' super directory, since this file should be stored in 'pymdp/tests/matlab_crossval/generation'

rng(7); % ensure the saved output file for `pymdp` is always the same
%% VARIABLE NAMES

T = 10; % total length of time (generative process horizon)
window_len = 3; % length of inference window (in the past)
policy_horizon = 1; % temporal horizon of policies
tau = 0.25; % learning rate
num_iter = 10; % number of variational iterations
num_states = [3]; % hidden state dimensionalities
num_factors = length(num_states); % number of hidden state factors
num_obs = [4];   % observation modality dimensionalities
num_modalities = length(num_obs); % number of hidden state factors
num_actions = [3]; % control factor (action) dimensionalities
num_control = length(num_actions);
w = 16.0; % equivalent of 'gamma' parameter in pymdp
 
qs_ppd = cell(1, num_factors); % variable to store posterior predictive density for current timestep. cell array of length num_factors, where each qs_ppd{f} is the PPD for a given factor (length [num_states(f), 1])
qs_bma = cell(1, num_factors); % variable to store bayesian model average for current timestep. cell array of length num_factors, where each xq{f} is the BMA for a given factor (length [num_states(f), 1])

states = zeros(num_factors,T); % matrix of true hidden states (separated by factor and timepoint) -- size(states) == [num_factors, T]
for f = 1:num_factors
    states(f,1) = randi(num_states(f));
end

actions = zeros(num_control, T); % history of actions along each control state factor and timestep -- size(actions) == [num_factors, T]
obs = zeros(num_modalities,T); % history of observations (separated by modality and timepoint) -- size (obs) == [num_modalities, T]
vector_obs = cell(num_modalities,T); % history of observations expressed as one-hot vectors


% allowable policies (here, specified as the next action) U
%--------------------------------------------------------------------------
Np        = num_actions;
U         = ones(1,Np,num_factors);
U(:,:,1)  = 1:Np;

policy_matrix = zeros(policy_horizon, Np, num_factors); % matrix of policies expressed in terms of time points, actions, and hidden state factors. size(policies) ==  [policy_horizon, num_policies, num_factors]. 
                                                                  % This gets updated over time with the actual actions/policies taken in the past

policy_matrix(1,:,:) = U;

p = 1:Np;

% likelihoods and priors

A = cell(1,num_modalities); % generative process observation likelihood (cell array of length num_modalities -- each A{g} is a matrix of size [num_modalities(g), num_states(:)]
B = cell(1,num_factors); % generative process transition likelihood (cell array of length num_factors -- each B{f} is a matrix of size [num_states(f), num_states(f), num_actions(f)]
C = cell(1,num_modalities);
lnC = cell(1, num_modalities);
for g= 1:num_modalities
    C{g} = repmat(rand(num_obs(g),1),1,T);
    lnC{g} = spm_log(spm_softmax(C{g}));
end

D = cell(1,num_factors); % prior over hidden states -- a cell array of size [1, num_factors] where each D{f} is a vector of length [num_states(f), 1]
for f = 1:num_factors
    D{f} = ones(num_states(f),1)/num_states(f);
end



for g = 1:num_modalities
    A{g} = spm_norm(rand([num_obs(g),num_states]));
end

a = A; % generative model == generative process


for f = 1:num_factors
    B{f} = spm_norm(rand(num_states(f), num_states(f), num_actions(f)));
end


b = B; % generative model transition likelihood (cell array of length num_factors -- each b{f} is a matrix of size [num_states(f), num_states(f), num_actions(f)]
b_t = cell(1,num_factors);

for f = 1:num_factors
    for u = 1:num_actions(f)
        b_t{f}(:,:,u) = spm_norm(b{f}(:,:,u)');% transpose of generative model transition likelihood
    end
end

for f = 1:num_factors
    Nu(f) = size(B{f},3);     % number of hidden controls
end
%% INITIALIZATION of beliefs


% initialise different posterior beliefs used in message passing

for f = 1:num_factors
    
    xn{f} = zeros(num_iter,num_states(f),window_len,T,Np);
    
    vn{f} = zeros(num_iter,num_states(f),window_len,T,Np);
    
    x{f}  = zeros(num_states(f),T,Np) + 1/num_states(f);
    
    d_policy_sep{f} = zeros(num_states(f),Np) + 1/num_states(f);

    qs_ppd{f}  = zeros(num_states(f), T, Np)      + 1/num_states(f);
    
    qs_bma{f}  = repmat(D{f},1,T);
    
    for k = 1:Np
        x{f}(:,1,k) = D{f};
        qs_ppd{f}(:,1,k) = D{f};
    end
    
end

%%

for t = 1:T
    
 
    % posterior predictive density over hidden (external) states
    %--------------------------------------------------------------
    for f = 1:num_factors       
        % Bayesian model average (xq)
        %----------------------------------------------------------
        xq{f} =  qs_bma{f}(:,t);       
    end
    
    % sample state, if not specified
    %--------------------------------------------------------------
    for f = 1:num_factors

        % the next state is generated by action on external states
        %----------------------------------------------------------

        if t > 1
            ps = B{f}(:,states(f,t - 1),actions(f,t - 1));
        else
            ps =  D{f};
        end
        states(f,t) = find(rand < cumsum(ps),1);

    end
    
    % sample observations, if not specified
    %--------------------------------------------------------------
    for g = 1:num_modalities
        
        % if observation is not given
        %----------------------------------------------------------
        if ~obs(g,t)
            % sample from likelihood given hidden state
            %--------------------------------------------------
            ind           = num2cell(states(:,t));
            p_obs            = A{g}(:,ind{:}); % gets the probability over observations, under the current hidden state configuration
            obs(g,t) = find(rand < cumsum(p_obs),1);
            vector_obs{g,t} = sparse(obs(g,t),1,1,num_obs(g),1);
        end
        
    end
    
    % Likelihood of observation under the various configurations of hidden states
    %==================================================================
    L{t} = 1;
    for g = 1:num_modalities
        L{t} = L{t}.*spm_dot(a{g},vector_obs{g,t});
    end
    
%     lh_seq = L(max(1,t-window_len+1):end); % prune the likelihood sequence to only carry those likelihoods relevant to the current moment
   
    % reset
    %--------------------------------------------------------------
    for f = 1:num_factors
        x{f} = spm_softmax(spm_log(x{f})/4);
    end
    
    S     = size(policy_matrix,1) + 1;   % horizon
    R = t;
    
    fprintf('Observations used for timestep %d\n',t)
    fprintf('===================================\n')
                        
    F     = zeros(Np,1);
    for k = p                % loop over plausible policies
%     dF    = 1;                  % reset criterion for this policy
        for iter = 1:num_iter       % iterate belief updates
            F(k)  = 0;                 % reset free energy for this policy           
            
            for j = max(1,t-window_len+1):S             % loop over future time points
                
                if t == 4 && j == S
                    debug_flag = true;
                end
                % curent posterior over outcome factors
                %--------------------------------------------------
                if j <= t
                    for f = 1:num_factors
                        xq{f} = x{f}(:,j,k);
                    end
                end

                for f = 1:num_factors

                    % hidden states for this time and policy
                    %----------------------------------------------
                    sx = x{f}(:,j,k);
                    qL = zeros(num_states(f),1);
                    v  = zeros(num_states(f),1);

                    % evaluate free energy and gradients (v = dFdx)
                    %----------------------------------------------
        %                 if dF > exp(-8) || iter > 4

                    % marginal likelihood over outcome factors
                    %------------------------------------------                   
                    
                    if j <= t
                        if iter == 1 && f == 1
                            fprintf('obs from timestep: %d\n',j)
                        end
                        qL = spm_dot(L{j},xq,f);
                        qL = spm_log(qL(:));
                    end                              

                    % entropy
                    %------------------------------------------
                    qx  = spm_log(sx);

                    % emprical priors (forward messages)
                    %------------------------------------------
                    if j == 1
                        px = spm_log(D{f});
                    elseif j == (t-window_len+1)                   
%                         px = spm_log(b{f}(:,:,policy_matrix(j - 1,k,f))*d_policy_sep{f}(:,k)); % policy separated prior
                        px = spm_log(d_policy_sep{f}(:,k)); % policy separated prior
                    else
                        px = spm_log(b{f}(:,:,policy_matrix(j - 1,k,f))*x{f}(:,j - 1,k));
                    end    
                    
                    v  = v + px + qL - qx;


                    % emprical priors (backward messages)
                    %------------------------------------------
                    if j < R
                        px = spm_log( b_t{f}(:,:,policy_matrix(j,k,f)) * x{f}(:,j+1,k) );
                        v  = v + px + qL - qx;
                    end

                    % (negative) free energy
                    %------------------------------------------
                    if j == 1 || j == S
                        F(k) = F(k) + sx'*0.5*v;
                    else
                        F(k) = F(k)  + sx'*(0.5*v - (num_factors-1)*qL/num_factors);
                    end

                    % update
                    %-----------------------------------------                

                    v    = v - mean(v);
 
                    sx   = softmax(qx + v * tau);

                    % store update neuronal activity
                    %----------------------------------------------
                    x{f}(:,j,k)          = sx;
                    xq{f}                = sx;
                    xn{f}(iter,:,j,t,k)  = sx;
                    vn{f}(iter,:,j,t,k)  = v;

                end
            end

            % convergence
            %------------------------------------------------------
%             if iter > 1
%                 dF = F - G;
%             end
%             G = F;

        end
    end
        
    %% expected free energy of policies
    
    S     = size(policy_matrix,1) + 1;   % horizon
    Q   = zeros(length(p),1);            % expected free energy for each policy

    for k = p
                    
        for j = t:S % loop over future timepoints (starting from t = current_t to t = (current_t + future_horizon)

            % get expected states for this policy and time
            %--------------------------------------------------
            for f = 1:num_factors
                xq{f} = x{f}(:,j,k);
            end

            % Bayesian surprise about states
            %--------------------------------------------------
            Q(k) = Q(k) + spm_MDP_G(A,xq);

            for g = 1:num_modalities

                % prior preferences about outcomes
                %----------------------------------------------
                qo   = spm_dot(A{g},xq);
                Q(k) = Q(k) + qo'*(lnC{g}(:,j));

            end
        end
    end
        
    % this is how the policy posterior is ultimately calculated in
    % original spm_MDP_VB_X:

    % previous expected precision
    %----------------------------------------------------------
%     if t > 1
%         w(t) = w(t - 1);
%     end
%     for i = 1:Ni
% 
%         % posterior and prior beliefs about policies
%         %------------------------------------------------------
%         qu = spm_softmax(qE + w(t)*Q + F);
%         pu = spm_softmax(qE + w(t)*Q);
% 
%         % precision (w) with free energy gradients (v = -dF/dw)
%         %------------------------------------------------------
%         if OPTIONS.gamma
%             w(t) = 1/beta;
%         else
%             eg      = (qu - pu)'*Q;
%             dFdg    = qb - beta + eg;
%             qb   = qb - dFdg/2;
%             w(t) = 1/qb;
%         end
% 
%         % simulated dopamine responses (expected precision)
%         %------------------------------------------------------
%         n             = (t - 1)*Ni + i;
%         wn(n,1)    = w(t);
%         un(:,n) = qu;
%         u(:,t)  = qu;
% 
%     end
   
    q_pi = spm_softmax(w*Q - F); % copied how we do it in `pymdp` for comparison
    
    disp(q_pi)
    
    if t < T
        % marginal posterior over action (for each factor)
        %----------------------------------------------------------
        Pu    = zeros(num_actions,1);
        for k = 1:length(p)
            sub        = num2cell(policy_matrix(t,k,:));
            Pu(sub{:}) = Pu(sub{:}) + q_pi(k);
        end
         
%         ind           = find(rand < cumsum(Pu(:)),1);
        ind           = find(Pu(:) == max(Pu(:)),1);
        actions(:,t) = spm_ind2sub(Nu,ind);
        
        for f = 1:num_factors
            policy_matrix(t,:,f) = actions(f,t);
        end
        for j = 1:size(U,1)
            if (t + 1) < T
                policy_matrix(t + 1,:,:) = U(:,:);
            end
        end
        
%         disp(size(policy_matrix,1))
        
        if (t - window_len) >= 0
            for f = 1:num_factors
                for k = 1:Np
                    d_policy_sep{f}(:,k) = x{f}(:,t-window_len+1,k);
                end
            end
%             policy_matrix = policy_matrix((t-window_len):end,:,:);
        end
        
        if t == (T-1)
            debug_flag = true;
        end
        
        % and re-initialise expectations about hidden states
        %------------------------------------------------------
        for f = 1:num_factors
            for k = 1:length(p)
                x{f}(:,:,k) = 1/num_states(f);
            end
        end
        
    end        
    
    if t == T
        obs  = obs(:,1:T);        % outcomes at 1,...,T
        states  = states(:,1:T);        % states   at 1,...,T
        actions  = actions(:,1:T - 1);    % actions  at 1,...,T - 1
        break;
    end
            
end

save_dir = 'output/vbx_test_1a.mat';
policies = U;
t_horizon = window_len;
qs = x;
likelihoods = L;
save(save_dir,'A','B','C','obs','states','actions','policies','t_horizon','actions','qs','xn', 'vn', 'likelihoods')
%%
% auxillary functions
%==========================================================================

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

function A  = spm_wnorm(A)
% summation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A   = A + 1e-16;
A   = bsxfun(@minus,1./sum(A,1),1./A)/2;
end

function sub = spm_ind2sub(siz,ndx)
% subscripts from linear index
%--------------------------------------------------------------------------
n = numel(siz);
k = [1 cumprod(siz(1:end-1))];
for i = n:-1:1
    vi       = rem(ndx - 1,k(i)) + 1;
    vj       = (ndx - vi)/k(i) + 1;
    sub(i,1) = vj;
    ndx      = vi;
end
end

function [X] = spm_dot(X,x,i)
% Multidimensional dot (inner) product
% FORMAT [Y] = spm_dot(X,x,[DIM])
%
% X   - numeric array
% x   - cell array of numeric vectors
% DIM - dimensions to omit (asumes ndims(X) = numel(x))
%
% Y  - inner product obtained by summing the products of X and x along DIM
%
% If DIM is not specified the leading dimensions of X are omitted.
% If x is a vector the inner product is over the leading dimension of X

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
% FORMAT [y] = spm_softmax(x,k)
%
% x - numeric array array
% k - precision, sensitivity or inverse temperature (default k = 1)
%
% y  = exp(k*x)/sum(exp(k*x))
%
% NB: If supplied with a matrix this routine will return the softmax
% function over colums - so that spm_softmax([x1,x2,..]) = [1,1,...]

% apply
%--------------------------------------------------------------------------
if nargin > 1,    x = k*x; end
if size(x,1) < 2; y = ones(size(x)); return, end

% exponentiate and normalise
%--------------------------------------------------------------------------
x  = exp(bsxfun(@minus,x,max(x)));
y  = bsxfun(@rdivide,x,sum(x));
end

function [G] = spm_MDP_G(A,x)

% preclude numerical overflow
%--------------------------------------------------------------------------
spm_log = @(x)log(x + exp(-16));

% probability distribution over the hidden causes: i.e., Q(x)
%--------------------------------------------------------------------------
qx    = spm_cross(x);

% accumulate expectation of entropy: i.e., E[lnP(o|x)]
%--------------------------------------------------------------------------
G     = 0;
qo    = 0;
for i = find(qx > exp(-16))'
    
    % probability over outcomes for this combination of causes
    %----------------------------------------------------------------------
    po   = 1;
    for g = 1:numel(A)
        po = spm_cross(po,A{g}(:,i));
    end
    po = po(:);
    qo = qo + qx(i)*po;
    G  = G  + qx(i)*po'*spm_log(po);
    
end

% subtract entropy of expectations: i.e., E[lnQ(o)]
%--------------------------------------------------------------------------
G  = G - qo'*spm_log(qo);


end

function [Y] = spm_cross(X,x,varargin)
% Multidimensional cross (outer) product
% FORMAT [Y] = spm_cross(X,x)
% FORMAT [Y] = spm_cross(X)
%
% X  - numeric array
% x  - numeric array
%
% Y  - outer product
%
% See also: spm_dot
%__________________________________________________________________________
% Copyright (C) 2015 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_cross.m 7527 2019-02-06 19:12:56Z karl $

% handle single inputs
%--------------------------------------------------------------------------
if nargin < 2
    if isnumeric(X)
        Y = X;
    else
        Y = spm_cross(X{:});
    end
    return
end

% handle cell arrays
%--------------------------------------------------------------------------
if iscell(X), X = spm_cross(X{:}); end
if iscell(x), x = spm_cross(x{:}); end

% outer product of first pair of arguments (using bsxfun)
%--------------------------------------------------------------------------
A = reshape(full(X),[size(X) ones(1,ndims(x))]);
B = reshape(full(x),[ones(1,ndims(X)) size(x)]);
Y = squeeze(bsxfun(@times,A,B));

% and handle remaining arguments
%--------------------------------------------------------------------------
for i = 1:numel(varargin)
    Y = spm_cross(Y,varargin{i});
end
end
