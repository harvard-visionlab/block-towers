
def run_samples(gen_fun, numBlocks, sideLength, std, truncate, numSamples):
    falls = []
    for i in range(numSamples):
        positions = gen_fun(numBlocks, sideLength, std, truncate=truncate)
        anyFall, isUnstable = compute_will_fall(positions, sideLength)
        falls.append(anyFall)

    return np.mean(falls)

def choose_variance(numBlocks, sideLength=.40, pFallTarget=.5, startStd=.29, truncate=.75, 
                    totalReversals=20, numSamples=1000, initial_step_size=.01):
    numReversals = 0
    direction = 0
    stds = []
    ps = []
    reversals = []
    currStd = startStd
    iter_num = 0
    samples = numSamples
    step_size = initial_step_size

    # Initialize progress bar
    pbar = progress_bar(range(totalReversals), display=True, leave=True)
    pbar.comment = 'Initializing'
    pbar.update(0)  # Update the progress bar for each reversal

    while numReversals < totalReversals:
        iter_num += 1

        if numReversals > totalReversals/2 and samples==numSamples:
            samples = numSamples * 2
            step_size = initial_step_size/2
        
        pFall = run_samples(numBlocks, sideLength, currStd, truncate, samples)
        stds.append(currStd)
        ps.append(pFall)
        pbar.comment = f"Starting iter {iter_num}, numReversals = {numReversals}, currStd = {currStd:.3f}, pFall = {pFall:.3f}"

        if (pFall < pFallTarget):
            currStd += step_size
            new_direction = 1
        else:
            currStd = currStd if (currStd - step_size) < 0 else (currStd - step_size)
            new_direction = -1


        if direction != new_direction and direction != 0:
            numReversals += 1
            reversals.append(True)
            pbar.update(numReversals)
        else:
            reversals.append(False)

        #pbar.update(0)
        #pbar.update(numReversals)
        direction = new_direction

    asymptote = np.array(stds)[reversals][-9:].mean()

    plt.plot(stds)
    plt.plot(ps)
    plt.legend(['std','pFall'])
    plt.title('For {} blocks, use std = {:2.3f}'.format(numBlocks, asymptote))

    return stds, ps, reversals, asymptote
