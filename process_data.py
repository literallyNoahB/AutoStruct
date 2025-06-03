'''
Handles automatic processing of file when uploaded to web-app
'''

# import dependencies
from ftir_proc_funcs import *

# Function to handle csv once loaded
def ProteinPipeline(df, sigma=13, std_guess=6, resilience=1, L=.048, title='', points_smooth=7, points_derive=7, spectrum_title='', upper_bound=1599, lower_bound=1710):
    # $ Step 1: 
    # isolate and baseline Amide I
    AmideI_df = FindAmideI(df, upper_bound=upper_bound, lower_bound=lower_bound)
    
    # $ Step 2:
    # normalize and perform FSD (and normalize again)
    normal_df = NormalizeSpectra(AmideI_df)
    FSD_df = GaussFSDSpect(normal_df, sigma, L=L)
    fsd_normal = NormalizeSpectra(FSD_df)
    
    # $ Step 3: 
    # second derivative and minima finding
    # using the smoothed/normalized non-fsd spectrum
    smooth_spec = SavGolSmooth(normal_df, points=points_smooth)
    savgol_second_deriv = SavGolDerive(smooth_spec, points=points_derive)
    minima_df = FindLocalMinimas(savgol_second_deriv, fsd_normal, resilience=resilience)
    
    # $ Step 4:
    # curve fitting - organize the parameters, curve fit
    params = OrganizeParams(minima_df, std_guess)
    guess_ys, guessParams, areas_df = NGuassianFit(params, fsd_normal)
    
    # $ Step 5:
    # calculate secondary structures
    content = AssignContent(areas_df)
    condensedContent = AssignCondensedContent(content)
    
    # $ Step 6:
    # return key values: image of whole spectrum, image of Amide I, non-condensed quantification, condensed quantification
    wavenums = fsd_normal.iloc[:, 0]
    intensities = fsd_normal.iloc[:, 1]
    # get the path to the saved processed gaussian member plot
    processed_graph_name = ColorCurves(wavenums, intensities, guess_ys, guessParams, title=title)
    # generate the path to the plot of the entire file the user sent (for quickly screening for any errors and completeness)
    whole_graph_name = GenWholeSpectrum(df, title=spectrum_title)
    
    # generate relevant csv files
    normalized_intensities = list(normal_df.iloc[:, 1])
    process_df, curve_df, content_df = CreateCSVs(wavenums, normalized_intensities, intensities, guess_ys, guessParams, content, condensedContent)
    
    return content, condensedContent, processed_graph_name, whole_graph_name, process_df, curve_df, content_df

# process information needed for producing the csv files
def CreateCSVs(wavenums, normalized_intensities, FSD_intensities, guess_ys, guessParams, content, condensedContent):
    # create csv of processed Amide I and fitted Amide I
    process_df = pd.DataFrame()
    process_df["Wavenumbers (cm^-1)"] = wavenums
    process_df["Normalized Amide I"] = normalized_intensities
    process_df["Fourier Self Deconvolution Intensities"] = FSD_intensities
    process_df["Regressed Intensities"] = guess_ys
    process_df.set_index("Wavenumbers (cm^-1)", inplace=True)
    
    # create csv of all curve properties - amp, loc, std
    amps = guessParams[::3]
    locs = guessParams[1::3]
    stds = guessParams[2::3]
    curve_df = pd.DataFrame()
    curve_df["Intensities"] = amps 
    curve_df["Locations (cm^-1)"] = locs
    curve_df["Standard Deviations (cm^-1)"] = stds
    
    # csv of relative contributions
    structure_names = np.array(list(content.keys()))
    structure_contents = np.array(list(content.values()))
    condensed_structure_names = np.array(list(condensedContent.keys()))
    condensed_structure_contents = np.array(list(condensedContent.values()))
    # add buffer lengths on condensed
    bufferLength = len(structure_contents) - len(condensed_structure_contents)
    buffer = np.empty(bufferLength)
    buffer[:] = np.nan 
    condensed_structure_names = np.append(condensed_structure_names, buffer)
    condensed_structure_contents = np.append(condensed_structure_contents, buffer)
    # construct the dataframe
    structure_df = pd.DataFrame()
    structure_df["Structure Name"] = structure_names
    structure_df["Structure Content"] = structure_contents
    structure_df["Condensed Structure Name"] = condensed_structure_names
    structure_df["Condensed Structure Contents"] = condensed_structure_contents
    
    # return all the dfs
    return process_df, curve_df, structure_df

# additional processing for nicely structuring the percentage content
def ProcessStructures(structure, condensed_structure):
    structure_names = list(structure.keys())
    structure_contents = [f"{i:.3%}" for i in structure.values()]
    cond_structure_names = list(condensed_structure.keys())
    cond_structure_contents = [f"{i:.3%}" for i in condensed_structure.values()]
    return structure_names, structure_contents, cond_structure_names, cond_structure_contents
    
    
    