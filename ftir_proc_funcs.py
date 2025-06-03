'''
All helper functions required to make the processing pipeline
This is the lowest level of control
'''

# ! import dependencies
# basic data science and csv manipulation
import pandas as pd
import numpy as np

# graphing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

import matplotlib
matplotlib.use('agg')

# spectra smoothing
from scipy.signal import savgol_filter

# deconvolution
from scipy.fft import fft
from scipy.fft import ifft
from scipy.fft import fftfreq

# curve fitting
from scipy.optimize import minimize

# generating file names
import secrets
import os

# ! Functions related to finding Amide I region
# upper minima finder
def FindUpperMinima(df, upper_bound):
    # rename df for simpler indexing
    df.columns = ["wavenumber", "peak"]

    # find higher minima - WORKS
    temp_df = df.loc[df["wavenumber"].between(upper_bound, upper_bound+1)] # previously 1575, 1625
    upperMinima_index = np.argmin(temp_df.loc[:, "peak"])+len(df.loc[df["wavenumber"] < upper_bound])

    return upperMinima_index

# lower minima finder
# also finds the Amide I region, and the correction parameters for baselining (linear substraction)
def FindLowerMinima(df, upperMinima, lower_bound):
  # rename columns for ease
  df.columns = ["wavenumber", "peak"]

  # define a subdf from 1690 to 1710
  temp_df = df.loc[df["wavenumber"].between(lower_bound, lower_bound+1)] # previously 1690, 1710

  # find absolute starting position of the temp_df
  # find value closest to 1690
  closestVal_index = np.argmin(np.abs(df.loc[:, "wavenumber"]-lower_bound)) # think the correct variable should be df

  # if the actual val is less than 1690, then good. else, subtract 1 from the index
  if df.iloc[closestVal_index, 0] < lower_bound:
    absoluteStartIndex = closestVal_index + 1
  else:
    absoluteStartIndex = closestVal_index

  # define tracking vars - tracked through indices
  position = -1
  trueMinima = 0
  ceiling = len(temp_df)-1
  complete = False

  # start iterating through and tacking vars
  while not complete:
    # step the position - done
    position += 1

    # regress line and check negative; if negative set the complete to True
    # doing this before the checking if the value is lower than the current minima avoids a potential error
    # find key points within the data drame
    # regress_df = df.iloc[position+absoluteStartIndex:upperMinima, :]
    regress_df = df.iloc[upperMinima:position+absoluteStartIndex, :]
    p0 = list(regress_df.iloc[-1, :])
    p1 = list(regress_df.iloc[0, :])
    # get x and y values and organize
    xVals = [p0[0], p1[0]]
    yVals = [p0[1], p1[1]]
    A = np.vstack([xVals, np.ones(len(xVals))]).T
    # perform the regression
    m, c = np.linalg.lstsq(A, yVals, rcond=None)[0]
    # perform the subtraction
    func = lambda xVals, m, c: m*xVals + c
    xVals = regress_df.loc[:, "wavenumber"]
    subtraction_vals = func(xVals, m, c)
    new_yVals = regress_df.loc[:, "peak"] - subtraction_vals

    # check if any negatives occur
    # we can't check the ends values since there's a chance through the regression algorithm the values will be considered
    # therefore we can only check between the 1 to -2 index
    if not all([i > 0 for i in new_yVals[1:-1]]):
      complete = True
      position -= 1 # the current position is not valid, it should be one lower

    # check if minima lower (condition 2)
    if temp_df.loc[:, "peak"].iloc[position] < temp_df.loc[:, "peak"].iloc[trueMinima]:
      trueMinima = position

    # if hit ceiling length (the back end of the wavenumber region has been hit)
    if position == ceiling:
      complete = True

  # if the while loop is complete, check to ensure that the last point is the lowest point
  # program uses the lowest minima which fits the no negative number constraint
  if position != trueMinima:
    position = trueMinima

  # isolate lower minima in the context of all other points
  lowerMinima = position + absoluteStartIndex

  # there's a lot of unique conditions for ending the for loop, so I'm going to
  # regress the line one more time, perform subtraction and return the processed df
  regress_df = df.iloc[upperMinima:lowerMinima, :]
  p0 = list(regress_df.iloc[-1, :])
  p1 = list(regress_df.iloc[0, :])
  # get x and y values
  xVals = [p0[0], p1[0]]
  yVals = [p0[1], p1[1]]
  A = np.vstack([xVals, np.ones(len(xVals))]).T
  # perform the regression
  m, c = np.linalg.lstsq(A, yVals, rcond=None)[0]
  # perform the subtraction
  func = lambda xVals, m, c: m*xVals + c
  xVals = regress_df.loc[:, "wavenumber"]
  subtraction_vals = func(xVals, m, c)
  new_yVals = regress_df.loc[:, "peak"] - subtraction_vals
  # organize final df
  corrected_df = regress_df.copy()
  corrected_df.loc[:, "peak"] = new_yVals

  return (corrected_df, lowerMinima, m, c)

# coordinate the upper and lower minima finders
def FindAmideI(df, upper_bound=1599, lower_bound=1710):
    upperMinima_index = FindUpperMinima(df, upper_bound)
    corrected_df, _, _, _ = FindLowerMinima(df, upperMinima_index, lower_bound)
    return corrected_df

# ! Normalization function
def NormalizeSpectra(df, normalizedHeight=1, min=0):
  # rename columns
  df.columns = ["wavenumber", "peak"]

  # shift the spectra up by the minima (so the base is min)
  minHeight = np.min(df.loc[:, "peak"])
  shiftHeight = minHeight - min
  df.loc[:, "peak"] = df.loc[:, "peak"] - shiftHeight

  # find the maxiumum height
  maxHeight = np.max(df.loc[:, "peak"])

  # find the factor to divide every height by
  divFactor = (maxHeight / normalizedHeight)

  # divide the heights
  normalHeights = df.loc[:, "peak"] / divFactor

  # return the spectra
  normal_df = pd.DataFrame()
  normal_df["wavenumber"] = df.loc[:, "wavenumber"]
  normal_df["normalized height"] = normalHeights
  return normal_df

# ! Functions related to Gaussian FSD
# apodization function
def ApodizationFunc(x, L):
  x = np.asarray(x)
  # Compute the piecewise function
  result = np.where(np.abs(x) <= L, 1 - np.abs(x) / L, 0)
  return result

# exponential function
def GaussExpFunc(xs, sigma, L):
  deconv = []
  for x in xs:
    if abs(x) <= L:
      val = np.exp(2*(np.pi*sigma*x)**2)
      deconv.append(val)
    else:
      deconv.append(0)
      
  return deconv
  
def GaussFSDSpect(spectra, sigma, L=.05):
  # decompose the spectrum
  intensity = np.array(spectra.iloc[:, 1])
  frequencies = np.array(spectra.iloc[:, 0])
  res = frequencies[1] - frequencies[0]

  # perform ifft
  interferogram = ifft(intensity)

  # get the interferogram x axis
  distances = fftfreq(len(frequencies), res)

  # apply apodization and exponential function
  apod = ApodizationFunc(distances, L)
  expFunc = GaussExpFunc(distances, sigma, L)
  modInterferogram = interferogram * expFunc * apod**2
  
  # transform back and return
  modSpec = fft(modInterferogram)

  # reform the df (better for downstream processing)
  FSD_df = pd.DataFrame()
  FSD_df["wavenumbers"] = frequencies
  FSD_df["peak"] = modSpec.real
  return FSD_df

# ! Smoothing function
def SavGolSmooth(df, points=7, polOrder=8, derivOrder=0):
  # rename columns
  df.columns = ["wavenumber", "peak"]

  # get the number of points to include in each window
  windowSize = round(len(df)/points)

  # perform the smoothing
  smoothedSpectra = savgol_filter(df.loc[:, "peak"], window_length=windowSize, polyorder=polOrder, deriv=derivOrder)

  # reassemble df
  smooth_df = pd.DataFrame()
  smooth_df["wavenumber"] = df["wavenumber"]
  smooth_df["peak"] = smoothedSpectra

  # return
  return smooth_df

# ! Second derivative function
def SavGolDerive(df, points=11, polOrder=8, derivOrder=2):
  # rename columns
  df.columns = ["wavenumber", "peak"]

  # get the number of points to include in each window
  windowSize = round(len(df)/points)

  # perform the smoothing
  derivedSpectra = savgol_filter(df.loc[:, "peak"], window_length=windowSize, polyorder=polOrder, deriv=derivOrder)

  # reassemble df
  deriv_df = pd.DataFrame()
  deriv_df["wavenumber"] = df["wavenumber"]
  deriv_df["peak"] = derivedSpectra

  # return
  return deriv_df

# ! Find relevant minimas
def FindLocalMinimas(derivSpectra, spectra, resilience=1):
  # rename columns
  derivSpectra.columns = ["wavenumber", "peak"]

  # create a storage system
  minima_df = pd.DataFrame(columns = ["wavenumber", "height"])

  # define initial window
  windowEnd = (2*resilience) + 1 # the interesting number is in the center of the window
  window = [0, windowEnd]

  # slide across region
  numberSlides = len(derivSpectra) - window[1]+1
  for i in range(numberSlides):
    # get the region(s)
    # region is the second derivative spectra to identify maximas
    # height region is the fsd normal spectrum so the initial height geusses can be obtained
    region = np.array(derivSpectra.iloc[window[0]:window[-1], :])
    heightRegion = np.array(spectra.iloc[window[0]:window[-1], :])
    # compare center to other elements
    if np.argmin(region[:, 1]) == resilience: # if it's a minima store it
      appendRow = [region[resilience, 0], heightRegion[resilience, 1]]
      minima_df.loc[len(minima_df.index)] = appendRow
    else: # if it isn't a minima do nothing
      pass
    # step the window region
    window = [i+1 for i in window]

  # return
  return minima_df

# ! Curve fitting
# organize parameters
# [intensity1, wavenumber1, std1, intensity2, wavenumber2, ...]
def OrganizeParams(minima_df, std_guess):
  wavenums = minima_df.iloc[:, 0]
  intensity = minima_df.iloc[:, 1]
  params = []
  for (wave, height) in zip(wavenums, intensity):
    params.append(height)
    params.append(wave)
    params.append(std_guess)

  return params

# perform curve fitting
def multiple_gaussians(x, params):
    y_total = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amplitude = params[i]
        location = params[i+1]
        std_deviation = params[i+2]
        y_total += amplitude * np.exp(-((x - location)**2) / (2 * std_deviation**2))
    return y_total

# Cost function to minimize
def cost_function(params, x, y):
    return np.sum((y - multiple_gaussians(x, params))**2)

# Callback function to capture intermediate results
def callback(params):
    intermediate_params.append(params)

# gaussian curve function
Gauss = lambda height, center, sigma, x: (height*np.exp(-(x-center)**2/(2*(sigma**2))))

# reconstruct aggregated curve (complete spectrum) from parameters
def ReconstructParams(params, x):
  # y var
  ys = np.zeros(len(x))

  # params are organized amplitude, wavenumber, standard deviation
  for (amp, wn, std) in zip(params[::3], params[1::3], params[2::3]):
    ys += Gauss(amp, wn, std, x)

  return ys

# integrate individual gaussian peak
def IntegrateGaussian(height, std):
    areaFactor = np.sqrt(2*np.pi)
    return areaFactor * std * height

# get area
def AreaDeconvCurves(params, x): # x may be an unecessary parameter depending on the gaussian integration functions
  # store curve parameters in entries in a dictionary
  paramDict = {}
  for index, curve in enumerate(range(0,int(len(params)),3)):
    curveParams = params[curve:curve+3]
    paramDict[index] = curveParams

  # create a area dict where the areas of curves are stored
  areaDict = {}
  for index, paramEntry in enumerate(list(paramDict.values())):
    areaDict[index] = IntegrateGaussian(paramEntry[0], paramEntry[2])

  return areaDict

# coordinate the proximity fit
def NGuassianFit(params, df):
    # get key values
    xs = df.iloc[:, 0]
    ys = df.iloc[:, 1]
    lowerMinima = df.iloc[0, 0]
    upperMinima = df.iloc[-1, 0]
    
    # do initial fit
    bounds=[(0, None), (lowerMinima, upperMinima), (0, None)] * (len(params) // 3) # amp, loc, std x n - allows the location to shift
      
    global intermediate_params
    intermediate_params = []
    result = minimize(cost_function, params, args=(xs, ys), callback=callback, method='SLSQP', bounds=bounds)
    # find the parameters to use
    if result.success == False:
        guessParams = intermediate_params[-1]
    else:
        guessParams = result.x
    
    # get the area and y values of function
    guess_ys = ReconstructParams(guessParams, xs)
    areas = AreaDeconvCurves(guessParams, xs)
    totalArea = sum(list(areas.values()))
    areas_df = pd.DataFrame()
    areas_df["area"] = areas
    areas_df["relative area"] = np.array(list(areas.values()))/totalArea
    areas_df["location"] = params[1::3]
        
    return guess_ys, guessParams, areas_df

# ! Assign content and condensed content
def AssignContent(areas_df):
  # assign regions
  # floor is exclusive, ceiling is inclusive

  structures = {
      "Side Chains/Aggregated Strands": (1605, 1616),
      "Aggregate Beta-strand/beta sheets (weak)": (1616, 1622),
      "Beta Sheets (strong)": (1622, 1638),
      "Random Coils/Extended Chains": (1638, 1647),
      "Random Coils": (1647, 1656),
      "Alpha Helices": (1656, 1663),
      "Turns": (1663, 1697),
      "Beta Sheets (weak)": (1697, 1704)
  }
  # structures = {
  #     "Side Chains": (1590, 1605),
  #     "Beta Sheet": (1610, 1635),
  #     "Random Coil": (1635, 1645),
  #     "Beta Turn": (1647, 1654),
  #     "Alpha Helix": (1658, 1664),
  #     "Turns and Bends": (1666, 1695),
  #     "Beta Sheet2": (1695, 1700)
  # }
  # content = {
  #     "Side Chains": 0,
  #     "Beta Sheet": 0,
  #     "Random Coil": 0,
  #     "Beta Turn": 0,
  #     "Alpha Helix": 0,
  #     "Turns and Bends": 0,
  #     "Beta Sheet2": 0
  # }

  content = {
      "Side Chains/Aggregated Strands": 0,
      "Aggregate Beta-strand/beta sheets (weak)": 0,
      "Beta Sheets (strong)": 0,
      "Random Coils/Extended Chains": 0,
      "Random Coils": 0,
      "Alpha Helices": 0,
      "Turns": 0,
      "Beta Sheets (weak)": 0
  }

  # iterate through each block and add area to content library
  totalArea = 0
  for i, row in areas_df.iterrows():
    j = list(row)
    for struct, (start, end) in structures.items():
        if start < j[2] <= end:
            content[struct] += j[0]
            totalArea += j[0]

  # normalize all the content to be a proportion of 1 (total)
  for struct in content:
    content[struct] = (content[struct] / totalArea)

  # return
  return content

# displaying structural information
def AssignCondensedContent(content, printStruct=False):
  # print whole dictionary
  if printStruct:
    print("Uncondensed Secondary Structure Content:")
    for key, value in content.items():
        print(f"{key}: {value * 100:.2f}%")

  # condense the dictionary and display
  # indices in the content structure and simplified structure listed below
  # index 0 is not included in side chains because it refers to specific aminos, and not secondary structure
  # since aggregated strands are not included, need to renormalize the secondary structure content
  # amorphous = 3, 4
  # beta structure = 1, 2, 7
  # alpha helices = 5
  # turns = 6
  contentNums = list(content.values())
  normalizationConst = sum(contentNums[1:])
  amorphousContent = (contentNums[3] + contentNums[4]) / normalizationConst
  betaStructureContent = (contentNums[1] + contentNums[2] + contentNums[7]) / normalizationConst
  alphaHelicesContent = (contentNums[5]) / normalizationConst
  turnsContent = (contentNums[6]) / normalizationConst
  condensedContent = {
      "Amorphous": amorphousContent,
      "Beta Structures": betaStructureContent,
      "Alpha Helices": alphaHelicesContent,
      "Turns": turnsContent
  }
  if printStruct:
    print("\nCondensed Secondary Structure Content (Simplified):")
    for key, value in condensedContent.items():
        print(f"{key}: {value * 100:.2f}%")

  # return the condensed dictionary
  return condensedContent

# ! Generate relevant graphs
def ColorCurves(fsdWavenums, fsdSpect, guess_ys2, guessparams2, title=''):
    # define figure elements
    fig, ax = plt.subplots(figsize=(8,6))

    # plot the true line
    ax.plot(fsdWavenums, fsdSpect, label="True FSD Spectrum", c='k', linewidth=2)

    # plot the second fit
    ax.plot(fsdWavenums, guess_ys2, label="Regressed Curve", c='tab:purple', linewidth=2)

    # define regions
    # Beta Structures = blue
    # Amorphous = green
    # Alpha Helices = yellow
    # Turns = orange
    structures = {
        "Amorphous 2": (1638, 1656),
        "Beta Structures 1": (1616, 1638),
        "Beta Structures 2": (1697, 1704),
        "Alpha Helices": (1656, 1663),
        "Turns": (1663, 1697)
    }

    # plot the constituent curves
    heights = guessparams2[::3]
    locations = guessparams2[1::3]
    stds = guessparams2[2::3]
    for (height, loc, std) in zip(heights, locations, stds):
        # assign the color
        peakColor = "none"
        for structure, (start, end) in structures.items():
            if start < loc <= end:
                if "Beta Structures" in structure:
                    # peakColor = "tab:blue" Here
                    peakColor = "cornflowerblue"
                elif "Amorphous" in structure:
                    peakColor = "tab:green"
                elif structure == "Alpha Helices":
                    peakColor = "gold"
                elif structure == "Turns":
                    peakColor = "tab:orange"

            # plot the curve
            # if there's peak that's unassigned, it should still be displayed
            if peakColor == "none":
                curve_ys = Gauss(height, loc, std, fsdWavenums)
                ax.plot(fsdWavenums, curve_ys, c='tab:olive')
                ax.fill_between(fsdWavenums, curve_ys, y2=0, alpha=.3, color='tab:olive')
            else:
                curve_ys = Gauss(height, loc, std, fsdWavenums)
                ax.plot(fsdWavenums, curve_ys, c=peakColor)
                ax.fill_between(fsdWavenums, curve_ys, y2=0, alpha=.3, color=peakColor)

    # set the legend
    black_patch = mpatches.Patch(color='k', label='FSD Spectrum')
    purple_patch = mpatches.Patch(color='tab:purple', label='Fitted Spectrum')
    blue_patch = mpatches.Patch(color='tab:blue', label='Beta Structures')
    green_patch = mpatches.Patch(color='tab:green', label='Amorphous')
    yellow_patch = mpatches.Patch(color='gold', label='Alpha Helices')
    orange_patch = mpatches.Patch(color='tab:orange', label='Turns')
    olive_patch = mpatches.Patch(color='tab:olive', label='Unassigned')

    plt.legend(handles=[black_patch, purple_patch, blue_patch, green_patch, yellow_patch, orange_patch, olive_patch],
                bbox_to_anchor=(1.28, 1.01))

    # set axes and title
    if title != "":
      ax.set_title(f"{title}", fontsize=14)
      

    # ax.set_xlabel("Wavenumber (cm^-1)", fontsize=13)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=13)
    ax.set_ylabel("Intensity", fontsize=13)

    ax.invert_xaxis()

    #   plt.show()
    # save the figure to user files
    # generate secret key for the name
    key_name = secrets.token_urlsafe(16)
    file_name = f"processed_graph_{key_name}.png"
    full_path = os.path.join('user_images', file_name)
    fig.savefig(full_path, bbox_inches='tight', dpi=300)
    return file_name

  
# generate whole spectrum
def GenWholeSpectrum(df, title=''):
    # extract wavenumbers and intensities
    wavenumbers = df.iloc[:, 0]
    intensities = df.iloc[:, 1]
    
    # construct graph
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(wavenumbers, intensities)
    
    if title != '':
      ax.set_title(f"{title}", fontsize=14)
    
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=13)
    ax.set_ylabel("Intensity", fontsize=13)
    
    ax.invert_xaxis()
    
    # save file and deal with path
    key_name = secrets.token_urlsafe(16)
    file_name = f"whole_graph_{key_name}.png"
    full_path = os.path.join('user_images', file_name)
    fig.savefig(full_path, bbox_inches='tight', dpi=300)
    
    # return
    return file_name
