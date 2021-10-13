%% 

clear all; close all; clc;

cd .. % this brings you into the 'pymdp/tests/matlab_crossval/' super directory, since this file should be stored in 'pymdp/tests/matlab_crossval/generation'

x     = linspace(1,32,128);
pA    = [1; 1];
rA    = pA;
rA(2) = 8;
F = zeros(numel(x), numel(x));
for i = 1:numel(x)
    for j = 1:numel(x)
        qA = [x(i);x(j)];
        F(i,j) = spm_MDP_log_evidence(qA,pA,rA);
    end
end

save_dir = 'output/bmr_test_a.mat';
save(save_dir, 'F');

%%
function [F,sA] = spm_MDP_log_evidence(qA,pA,rA)
% Bayesian model reduction for Dirichlet hyperparameters
% FORMAT [F,sA] = spm_MDP_log_evidence(qA,pA,rA)
%
% qA  - sufficient statistics of posterior of full model
% pA  - sufficient statistics of prior of full model
% rA  - sufficient statistics of prior of reduced model
%
% F   - free energy or (negative) log evidence of reduced model
% sA  - sufficient statistics of reduced posterior
%
% This routine computes the negative log evidence of a reduced model of a
% categorical distribution parameterised in terms of Dirichlet
% hyperparameters (i.e., concentration parameters encoding probabilities).
% It uses Bayesian model reduction to evaluate the evidence for models with
% and without a particular parameter.
% 
% It is assumed that all the inputs are column vectors.
%
% A demonstration of the implicit pruning can be found at the end of this
% routine
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_MDP_log_evidence.m 7326 2018-06-06 12:16:40Z karl $


% change in free energy or log model evidence
%--------------------------------------------------------------------------
sA = qA + rA - pA;
F  = spm_betaln(qA) + spm_betaln(rA) - spm_betaln(pA) - spm_betaln(sA);

end

function y = spm_betaln(z)
% returns the log the multivariate beta function of a vector.
% FORMAT y = spm_betaln(z)
%   y = spm_betaln(z) computes the natural logarithm of the beta function
%   for corresponding elements of the vector z. if concerned is an array,
%   the beta functions are taken over the elements of the first to mention
%   (and size(y,1) equals one).
%
%   See also BETAINC, BETA.
%--------------------------------------------------------------------------
%   Ref: Abramowitz & Stegun, Handbook of Mathematical Functions, sec. 6.2.
%   Copyright 1984-2004 The MathWorks, Inc. 
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_betaln.m 7508 2018-12-21 09:49:44Z thomas $

% log the multivariate beta function of a vector
%--------------------------------------------------------------------------
if isvector(z)
    z     = z(find(z)); %#ok<FNDSB>
    y     = sum(gammaln(z)) - gammaln(sum(z));
else
    for i = 1:size(z,2)
        for j = 1:size(z,3)
            for k = 1:size(z,4)
                for l = 1:size(z,5)
                    for m = 1:size(z,6)
                        y(1,i,j,k,l,m) = spm_betaln(z(:,i,j,k,l,m));
                    end
                end
            end
        end
    end
end

end



