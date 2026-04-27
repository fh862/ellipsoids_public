using System;
using System.IO;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AEPsych;

public class TrialHelper
{
    private System.Random rand; // Store the Random instance as a class variable

    // Constructor with an optional seed parameter
    public TrialHelper(int? seed = null)
    {
        // If a seed is provided, use it; otherwise, use a time-based seed
        rand = seed.HasValue ? new System.Random(seed.Value) : new System.Random();
    }

    // Generates a random integer indicating the location of the odd stimulus.
    // In the default case (nLocations = 4), the reference and comparison stimuli
    // are fully randomized among the three spatial positions (top, left, right).
    // When nLocations is set to 3, this typically indicates a suprathreshold experiment
    // where the reference stimulus is fixed in position, and only the comparison stimuli
    // are shuffled across the remaining locations.
    public int RefCompRandomSequence(int nLocations = 4)
    {
        return rand.Next(1, nLocations); // Generates a random number between 1 and (nLocations - 1)
    }

    // Shuffles [0,1,2] using the instance RNG and returns the order
    public List<int> ShuffleRefComp1Comp2()
    {
        var list = new List<int> { 0, 1, 2 };
        ShuffleInPlace(list);
        return list;
    }

    // Generic Fisher–Yates using the instance RNG
    private void ShuffleInPlace<T>(IList<T> list)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rand.Next(i + 1); // 0..i
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    // Decodes a shuffled assignment of stimuli to screen positions.
    // The input <paramref name="order"/> must be a permutation of {0,1,2}, where
    // the values encode the stimuli (0 = Ref, 1 = Comp1, 2 = Comp2) and the
    // indices encode the positions (index 0 = Top, 1 = Left, 2 = Right).
    // For example:
    //   order = [0,1,2] => Ref at Top (0), Comp1 at Left (1), Comp2 at Right (2)
    //   order = [2,0,1] => Ref at Left (1), Comp1 at Right (2), Comp2 at Top (0)
    // The method validates the input and throws if it is not a proper permutation.
    public (int RefLocation, int Comp1Location, int Comp2Location) DecodeLocations(IList<int> order)
    {
        int refLoc = order.IndexOf(0); // 0=Top, 1=Left, 2=Right
        int comp1Loc = order.IndexOf(1);
        int comp2Loc = order.IndexOf(2);

        return (refLoc, comp1Loc, comp2Loc);
    }
    
    // Method to check user response for the 3-alternative forced-choice (3AFC) oddity task
    // The observer selects the stimulus that looks different (odd one out)
    // OddLocation indicates the position of the odd stimulus (1 = top, 2 = left, 3 = right)
    // The method returns 1f if the observer's response matches the odd stimulus location, 0f otherwise
    public float CheckUserResponse(bool key0pressed, bool key1pressed, bool key2pressed, int OddLocation)
    {
        return ((OddLocation == 1 && key0pressed) ||
                (OddLocation == 2 && key1pressed) ||
                (OddLocation == 3 && key2pressed)) ? 1f : 0f;
    }

    // Method to check user response for a 2-alternative forced-choice (2AFC) suprathreshold task
    // where the reference stimulus is fixed and the comparison stimuli are on the left and right
    // CompLocation = 1 if the first comparison is on the left, 2 if it’s on the right
    // If the observer selects the first comparison (presses the key on the same side),
    // return 1f; otherwise return 2f, indicating the second comparison was selected
    public float CheckUserResponse_suprathres(bool key0pressed, bool key1pressed, int CompLocation)
    {
        return ((CompLocation == 1 && key0pressed) ||
                (CompLocation == 2 && key1pressed)) ? 1f : 2f;
    }

    // Method to scale the comparison value
    // NOTE: This function was used in the pilot experiment (interleaved 2D thresholding task), 
    // where trial placement was determined by the AEPsych Unity client. 
    // In the initial version, Sobol-generated trials were scaled by different factors ([1/4, 2/4, 3/4]) 
    // to balance task difficulty, and the scalers were shuffled accordingly.
    // In the updated experiment (4D thresholding task), this method is NO LONGER NEEDED because 
    // both shuffling and scaling are now handled on the Python side.
    public float ScaleComparison(float refValue, float compValue, float scaler)
    {
        // Scale the comparison value
        float scaledValue = (compValue - refValue) * scaler + refValue;

        // Debugging output
        //Debug.Log("Trial: " + trialCounter + ", Scaler: " + scaler +
        //          ", compValue: " + compValue + ", scaledValue: " + scaledValue);
        return scaledValue;
    }

    // Method to extract the MOCS trial sequence from a CSV file
    // NOTE: This function was used in the pilot experiment (interleaved 2D thresholding task),
    // where AEPsych and MOCS conditions were run in separate sessions. 
    // In the pilot version, MOCS trials were pre-generated in MATLAB, saved in CSV files, 
    // and then loaded into C# for execution.
    // In the updated experiment (4D thresholding task), this method is NO LONGER NEEDED because 
    // MOCS trials are now generated, shuffled, and interleaved with AEPsych trials on the Python side.
    public List<List<float>> LoadMOCSTrialSequence(string filePath, int trialsPerCondition)
    {
        List<List<float>> trialSequence = new List<List<float>>();

        try
        {
            // Check if the file exists before proceeding
            if (!File.Exists(filePath))
            {
                Debug.LogError($"Error: File not found: {filePath}");
                return trialSequence; // Return an empty list
            }

            Debug.Log($"File found: {filePath}");

            string[] lines = File.ReadAllLines(filePath);
            foreach (string line in lines)
            {
                string[] values = line.Split(',');
                float wDim1 = float.Parse(values[0]);
                float wDim2 = float.Parse(values[1]);
                trialSequence.Add(new List<float> { wDim1, wDim2 });
            }

            // Validate the number of trials
            int trialCount = trialSequence.Count;
            if (trialCount != trialsPerCondition)
            {
                Debug.LogError($"Error: File '{filePath}' does not contain {trialsPerCondition} trials. It contains {trialCount} trials.");
                throw new Exception($"MOCS trial sequence validation failed for file '{filePath}': It has {trialCount} trials instead of {trialsPerCondition}.");
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Error loading MOCS trial sequence: " + e.Message);
        }

        return trialSequence;
    }

}

public class FileDataExtractor
{
    // Coroutine to verify file existence and check if the last line matches expected content
    public IEnumerator VerifyFileExistsAndCheckNthLastCharacters(string filename, string expectedWord, int nthLastWord,
    float retryDelay = 1f / 60f, float maxWaitTime = 1200f)
    {
        float elapsedTime = 0f;

        // Wait for the file to exist
        while (!File.Exists(filename))
        {
            if (elapsedTime >= maxWaitTime)
            {
                Debug.LogWarning($"Timeout: File '{filename}' not found within {maxWaitTime} seconds.");
                yield break; // Exit the coroutine
            }

            yield return new WaitForSeconds(retryDelay);
            elapsedTime += retryDelay;
        }

        bool isMatch = false;
        elapsedTime = 0f; // Reset elapsed time for the second waiting loop

        // Wait until the expected word appears in the last line
        while (!isMatch)
        {
            string lastLine = ReadLastLine(filename);
            (isMatch, _) = CheckLastNthCharacters(lastLine, expectedWord, nthLastWord);

            if (isMatch) break; // Exit loop if match is found

            if (elapsedTime >= maxWaitTime)
            {
                Debug.LogWarning($"Timeout: Expected word '{expectedWord}' not found in '{filename}' within {maxWaitTime} seconds.");
                yield break; // Exit the coroutine
            }

            yield return new WaitForSeconds(retryDelay);
            elapsedTime += retryDelay;
        }
    }

    // Method to read the last line from a file safely
    public string ReadLastLine(string path)
    {
        string lastLine = string.Empty;

        try
        {
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (var sr = new StreamReader(fs))
            {
                while (!sr.EndOfStream)
                {
                    lastLine = sr.ReadLine();
                }
            }
        }
        catch (IOException)
        {
            Debug.Log("File is currently in use by another process. Will retry...");
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ Error reading file '{path}': {e.Message}");
        }

        return lastLine; // Returns an empty string if reading fails
    }

    // Method to check if the nth last word in a line matches an expected word
    public (bool, string) CheckLastNthCharacters(string line, string expectedWord, int n)
    {
        line = line.Trim();
        string[] words = line.Split(' ');

        if (n >= 0 && n < words.Length)
        {
            string nthLastWord = words[words.Length - n - 1];
            return (nthLastWord == expectedWord, nthLastWord);
        }
        return (false, string.Empty);
    }

    // Method to extract RGB values from a formatted string
    public (float, float, float) ExtractRGBValues(string colorString)
    {
        string[] parts = colorString.Split('_');

        string rString = parts[0].Substring(1);
        float r = float.Parse(rString);

        string gString = parts[1].Substring(1);
        float g = float.Parse(gString);

        string bString = parts[2].Substring(1);
        float b = float.Parse(bString);

        return (r, g, b);
    }

    public (string trialInfo, string refString, string compString) ExtractTrialRefComp(string input)
    {
        // Split the input string by spaces
        string[] parts = input.Split(' ');

        // Initialize variables to store the extracted values
        string trialInfo = null;
        string refString = null;
        string compString = null;

        // Iterate through each part and find the relevant values
        foreach (string part in parts)
        {
            if (part.StartsWith("Trial_"))
            {
                trialInfo = part; // Extract trial info
            }
            else if (part.StartsWith("Ref_"))
            {
                refString = part.Replace("Ref_", ""); // Remove "Ref_R" prefix
            }
            else if (part.StartsWith("Comp_"))
            {
                compString = part.Replace("Comp_", ""); // Remove "Comp_" prefix
            }
        }

        return (trialInfo, refString, compString);
    }

    public (string trialInfo, string refString, string compString, string comp2String) ExtractTrialRefComp2(string input)
    {
        // Split the input string by spaces
        string[] parts = input.Split(' ');

        // Initialize variables to store the extracted values
        string trialInfo = null;
        string refString = null;
        string compString = null;
        string comp2String = null;

        // Iterate through each part and find the relevant values
        foreach (string part in parts)
        {
            if (part.StartsWith("Trial_"))
            {
                trialInfo = part;
            }
            else if (part.StartsWith("Ref_"))
            {
                refString = part.Replace("Ref_", "");
            }
            else if (part.StartsWith("Comp_"))
            {
                compString = part.Replace("Comp_", "");
            }
            else if (part.StartsWith("Comp2_"))
            {
                comp2String = part.Replace("Comp2_", "");
            }
        }

        return (trialInfo, refString, compString, comp2String);
    }

    public (string trialInfo, string refString, string compString, string backgroundString) ExtractTrialRefCompBackground(string input)
    {
        // Split the input string by spaces
        string[] parts = input.Split(' ');

        // Initialize variables to store the extracted values
        string trialInfo = null;
        string refString = null;
        string compString = null;
        string backgroundString = null;

        // Iterate through each part and find the relevant values
        foreach (string part in parts)
        {
            if (part.StartsWith("Trial_"))
            {
                trialInfo = part;
            }
            else if (part.StartsWith("Ref_"))
            {
                refString = part.Replace("Ref_", "");
            }
            else if (part.StartsWith("Comp_"))
            {
                compString = part.Replace("Comp_", "");
            }
            else if (part.StartsWith("Background_"))
            {
                backgroundString = part.Replace("Background_", "");
            }
        }

        return (trialInfo, refString, compString, backgroundString);
    }

    public (string BackgroundString, string CubicRoomString) ExtractBackgroundCubicRoom(string input)
    {
        // Split the input string by spaces
        string[] parts = input.Split(' ');

        // Initialize variables to store the extracted values
        string BackgroundString = null;
        string CubicRoomString = null;

        // Iterate through each part and find the relevant values
        foreach (string part in parts)
        {
            if (part.StartsWith("Background_"))
            {
                BackgroundString = part.Replace("Background_", "");
            }
            else if (part.StartsWith("CubicRoom_"))
            {
                CubicRoomString = part.Replace("CubicRoom_", "");
            }
        }

        return (BackgroundString, CubicRoomString);
    }

    public string ConvertRGBToString(params float[] rgbValues)
    {
        return $"R{rgbValues[0]:F8}_G{rgbValues[1]:F8}_B{rgbValues[2]:F8}";
    }

    public string ConvertColorToString(Color c)
    {
        return $"R{c.r:F8}_G{c.g:F8}_B{c.b:F8}";
    }

    // Method to write a message with a timestamp to a specified file
    public void WriteMessageToFile(string filePath, string message)
    {
        string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        string fullMessage = $"{timestamp} - Unity: {message}";

        try
        {
            using (StreamWriter sw = new StreamWriter(filePath, true))
            {
                sw.WriteLine(fullMessage);
            }
        }
        catch (IOException e)
        {
            Debug.LogError($"Failed to write to file: {e.Message}");
        }
    }

    public void WriteResponseToFile(string initFilePath, string trialInfo, string refString, string compString, float response)
    {
        // Ensure Unity waits for file writing before proceeding
        WriteMessageToFile(initFilePath, $"{trialInfo} Ref_{refString} Comp_{compString} Resp{(int)response} Image_Confirmed");
    }
    
    public void WriteResponseToFileComp2(string initFilePath, string trialInfo, string refString, string compString, string comp2String, float response)
    {
        // Ensure Unity waits for file writing before proceeding
        WriteMessageToFile(initFilePath, $"{trialInfo} Ref_{refString} Comp_{compString} Comp2_{comp2String} Resp{(int)response} Image_Confirmed");
    }
}

// Separate class for gamma correction helper
public class GammaCorrectionHelper
{
    private string gammaCorrectionFile;
    private List<double> interpLuminance = new List<double>();
    private List<double> redCorrected = new List<double>();
    private List<double> greenCorrected = new List<double>();
    private List<double> blueCorrected = new List<double>();

    public GammaCorrectionHelper(string gammaCorrectionFile)
    {
        this.gammaCorrectionFile = gammaCorrectionFile;
    }

    // Public properties to access private fields
    public List<double> InterpLuminance => interpLuminance;
    public List<double> RedCorrected => redCorrected;
    public List<double> GreenCorrected => greenCorrected;
    public List<double> BlueCorrected => blueCorrected;

    // Method to load gamma correction data from CSV file
    public void LoadGammaCorrectionData()
    {
        // Ensure the file exists
        if (!File.Exists(gammaCorrectionFile))
        {
            Debug.LogError($"Gamma correction file not found: {gammaCorrectionFile}");
            return;
        }

        // Read all lines from the CSV file
        string[] lines = File.ReadAllLines(gammaCorrectionFile);

        // Skip the header line and parse each row
        for (int i = 1; i < lines.Length; i++)
        {
            string[] values = lines[i].Split(',');

            if (values.Length >= 4) // Ensure there are at least 4 columns
            {
                interpLuminance.Add(double.Parse(values[0]));
                redCorrected.Add(double.Parse(values[1]));
                greenCorrected.Add(double.Parse(values[2]));
                blueCorrected.Add(double.Parse(values[3]));
            }
        }
        
        if (interpLuminance.Count == 0 || redCorrected.Count == 0 || greenCorrected.Count == 0 || blueCorrected.Count == 0)
        {
            Debug.LogError("Gamma correction data not loaded.");
        }
        else
        {
            Debug.Log($"Gamma correction data loaded successfully from: {gammaCorrectionFile}");
        }
    }

    // Method to apply gamma correction by finding the closest index in interpLuminance
    public (double, double, double) GammaCorrect(double r, double g, double b)
    {
        // Find the closest index in interpLuminance for each color channel
        int rIndex = FindClosestIndex(r, interpLuminance);
        int gIndex = FindClosestIndex(g, interpLuminance);
        int bIndex = FindClosestIndex(b, interpLuminance);

        // Get the gamma-corrected values using the closest indices
        double correctedR = redCorrected[rIndex];
        double correctedG = greenCorrected[gIndex];
        double correctedB = blueCorrected[bIndex];

        return (correctedR, correctedG, correctedB);
    }

    private int FindClosestIndex(double value, List<double> list)
    {
        return list
            .Select((x, index) => new { Value = x, Index = index })  // Create an anonymous object with Value and Index
            .OrderBy(x => Math.Abs(x.Value - value))                 // Order by the absolute difference
            .First().Index;                                          // Get the index of the closest value
    }

}

// New class to handle surface color changing
public class SurfaceColorChanger
{
    private readonly GammaCorrectionHelper gammaCorrectionHelper;
    private readonly bool gammaCorrectionEnabled;
    private readonly float smoothness;

    public SurfaceColorChanger(GammaCorrectionHelper gammaCorrectionHelper, bool gammaCorrectionEnabled, float smoothness)
    {
        this.gammaCorrectionHelper = gammaCorrectionHelper;
        this.gammaCorrectionEnabled = gammaCorrectionEnabled;
        this.smoothness = smoothness;
    }

    public Color ChangeSurfaceColor(Renderer targetRenderer, float r, float g, float b)
    {
        Color newColor;

        if (gammaCorrectionEnabled && gammaCorrectionHelper != null)
        {
            // Example usage of gamma correction
            // using double for gamma-corrected values and then casting them to float
            // This is necessary because Unity’s Color struct expects float values (32-bit), not double (64-bit).
            (double correctedR, double correctedG, double correctedB) = gammaCorrectionHelper.GammaCorrect(r, g, b);
            // Create a new color from the corrected r, g, b values
            newColor = new Color((float)correctedR, (float)correctedG, (float)correctedB);
        }
        else
        {
            newColor = new Color(r, g, b);
        }
        // Set the color of the blob if the shader is set to Standard
        targetRenderer.material.color = newColor;
        // Set the smoothness of the material (0 = matte, 1 = glossy)
        targetRenderer.material.SetFloat("_Glossiness", smoothness);

        return newColor;
    }

    public Color ChangeSurfaceTextureColor(Renderer targetRenderer, float r, float g, float b)
    {
        Color newColor;

        if (gammaCorrectionEnabled && gammaCorrectionHelper != null)
        {
            // Example usage of gamma correction
            // using double for gamma-corrected values and then casting them to float
            // This is necessary because Unity’s Color struct expects float values (32-bit), not double (64-bit).
            (double correctedR, double correctedG, double correctedB) = gammaCorrectionHelper.GammaCorrect(r, g, b);
            // Create a new color from the corrected r, g, b values
            newColor = new Color((float)correctedR, (float)correctedG, (float)correctedB);
        }
        else
        {
            // Create a new color from the given r, g, b values
            newColor = new Color(r, g, b);
        }
        // Create a 1x1 texture to mimic a uniform color application
        Texture2D newTexture = new Texture2D(1, 1, TextureFormat.RGBAFloat, false);
        // Set the single pixel in the texture to the specified color
        newTexture.SetPixel(0, 0, newColor);
        newTexture.Apply();

        //Debug.Log("Texture applied with color: " + newColor);

        // Create a new material with the Standard shader   
        targetRenderer.material = new Material(Shader.Find("Standard"));
        // Assign the new 1x1 texture as the main texture of the material
        targetRenderer.material.mainTexture = newTexture;
        // Set the texture scale to ensure the single color covers the entire object
        targetRenderer.material.mainTextureScale = new Vector2(1, 1);
        targetRenderer.material.mainTextureOffset = new Vector2(0, 0);
        // Set the glossiness/smoothness of the material
        targetRenderer.material.SetFloat("_Glossiness", smoothness);

        return newColor;
    }

}

public class CameraBackgroundChanger
{
    private readonly GammaCorrectionHelper gammaCorrectionHelper;
    private readonly bool gammaCorrectionEnabled;

    public CameraBackgroundChanger(GammaCorrectionHelper gammaCorrectionHelper, bool gammaCorrectionEnabled)
    {
        this.gammaCorrectionHelper = gammaCorrectionHelper;
        this.gammaCorrectionEnabled = gammaCorrectionEnabled;
    }

    public Color ChangeCameraBackground(Camera camera, float r, float g, float b)
    {
        Color newColor;
        // Apply gamma correction if enabled
        if (gammaCorrectionEnabled && gammaCorrectionHelper != null)
        {
            (double correctedR, double correctedG, double correctedB) = gammaCorrectionHelper.GammaCorrect(r, g, b);
            newColor = new Color((float)correctedR, (float)correctedG, (float)correctedB);
        }
        else
        {
            newColor = new Color(r, g, b); // alpha defaults to 1f
        }
        // change the camera's background color
        camera.backgroundColor = newColor;  // use the corrected color
        return newColor;
    }
}


// Classes that capture screenshots, save exr files and text files
public class FrameBufferUtils
{
    public static void CaptureRenderFrameBuffer(Camera camera, string saveDirectory, int frameCounter)
    {
        // Create a RenderTexture with higher precision (8-bit per channel)
        RenderTexture renderTexture = new RenderTexture(Screen.width, Screen.height, 24, RenderTextureFormat.ARGB32);
        renderTexture.enableRandomWrite = true;
        renderTexture.Create();

        // Set the camera's target texture to the high-precision render texture
        camera.targetTexture = renderTexture;
        camera.Render();

        // Activate the RenderTexture
        RenderTexture.active = renderTexture;

        // Capture the frame buffer into a Texture2D with floating point precision
        Texture2D texture = new Texture2D(Screen.width, Screen.height, TextureFormat.RGBAHalf, false);
        texture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        texture.Apply();

        // Create a counter-based file name (e.g., FrameBufferImage_0001.exr)
        string fileName = $"FrameBufferImage_{frameCounter:D4}.exr"; // D4 formats the counter as 4 digits (e.g., 0001)
        string filePath = Path.Combine(saveDirectory, fileName);

        // Save the texture as an EXR file
        byte[] bytes = texture.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat);
        File.WriteAllBytes(filePath, bytes);

        Debug.Log($"Frame buffer saved to EXR: {filePath}");

        // Clean up
        camera.targetTexture = null;
        RenderTexture.active = null;
        UnityEngine.Object.Destroy(renderTexture);
    }

    public static void SaveRGB8BitToTxt(Texture2D texture, string saveDirectory, string fileName)
    {
        // Get the width and height of the texture
        int width = texture.width;
        int height = texture.height;

        // Construct the txt file path
        string txtFilePath = Path.Combine(saveDirectory, fileName);

        // Open the file for writing
        using (StreamWriter writer = new StreamWriter(txtFilePath, true))
        {
            // Iterate over the specific portion of the texture
            for (int y = 0; y < height; y++)  // Iterate through 2nd and 3rd rows
            {
                for (int x = 0; x < width; x++)  // Iterate through 2nd column
                {
                    // Get the pixel color at (x, y)
                    Color pixelColor = texture.GetPixel(x, y);

                    // Convert floating-point color values (0-1) to 8-bit integers (0-255)
                    int r8Bit = Mathf.RoundToInt(pixelColor.r * 255);
                    int g8Bit = Mathf.RoundToInt(pixelColor.g * 255);
                    int b8Bit = Mathf.RoundToInt(pixelColor.b * 255);

                    // Ensure values are within the 8-bit range (0-255) after rounding
                    r8Bit = Mathf.Clamp(r8Bit, 0, 255);
                    g8Bit = Mathf.Clamp(g8Bit, 0, 255);
                    b8Bit = Mathf.Clamp(b8Bit, 0, 255);

                    // Write the RGB values for each pixel to the txt file in plain text format
                    writer.WriteLine($"Pixel ({x},{y}) -> R: {r8Bit}, G: {g8Bit}, B: {b8Bit}");
                }
            }
        }
        Debug.Log($"8-bit RGB values saved to TXT: {txtFilePath}");
    }

    // Helper method to capture a screenshot on the first trial
    public static void Capture1Screenshot(string phase, string screenshotDirectory)
    {
        // Create the directory if it doesn't exist
        if (!Directory.Exists(screenshotDirectory))
        {
            Directory.CreateDirectory(screenshotDirectory);
        }

        // Set the full path for the screenshot file
        string screenshotFileName = Path.Combine(screenshotDirectory, $"Trial1_{phase}_{Time.frameCount}.png");

        // Capture and save the screenshot
        ScreenCapture.CaptureScreenshot(screenshotFileName);
        Debug.Log($"Screenshot captured: {screenshotFileName}");
    }
}


// NOTE: This entire class is now obsolete. It was originally used in the pilot experiment 
// (interleaved 2D thresholding task), where trial placement was controlled by the AEPsych Unity client.
// In the initial implementation, trial placement was determined by the AEPsych C# client, 
// requiring configuration files to define task-specific parameters (e.g., parameter bounds).
// In the updated experiment (4D thresholding task), this functionality is NO LONGER NEEDED 
// as trial placement, configuration, and task management are now handled on the Python side.
public class ConfigFileHelper
{
    private string configText;
    
    public List<string> StrategyNames { get; private set; }      // List of strategy names
    public List<int> MinimumAsks { get; private set; }           // List of minimum asks (e.g., 10, 10, 10, 330)
    public List<int> CumulativeAsks { get; private set; }        // List of cumulative asks (e.g., 10, 20, 30, 360)
    public List<float> SobolScalerList { get; private set; }     // List of Sobol scalers (e.g., 1.0, 0.5, 0.0, 1.0)
    public List<float> ParameterLowerBounds { get; private set; }// List of parameter lower bounds
    public List<float> ParameterUpperBounds { get; private set; }// List of parameter upper bounds
    public float Wdim1Ref { get; private set; }                  // Reference value for W dimension 1, bounded between -1 and 1
    public float Wdim2Ref { get; private set; }                  // Reference value for W dimension 2, bounded between -1 and 1

    public ConfigFileHelper(string configText)
    {
        this.configText = configText;
        StrategyNames = new List<string>();
        MinimumAsks = new List<int>();
        CumulativeAsks = new List<int>();
        SobolScalerList = new List<float>();
        ParameterLowerBounds = new List<float>();
        ParameterUpperBounds = new List<float>();
    }

    // Retrieve strategy names from the config file
    public void RetrieveStrategyNames()
    {
        string[] lines = configText.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);

        foreach (string line in lines)
        {
            if (line.StartsWith("strategy_names"))
            {
                string strategyList = line.Split('=')[1].Trim();
                strategyList = strategyList.Trim('[', ']');
                StrategyNames.AddRange(strategyList.Split(',').Select(s => s.Trim()));
                break;
            }
        }
    }

    // Retrieve minimum asks from the config file
    public void RetrieveMinimumAsks()
    {
        string[] lines = configText.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);

        foreach (string line in lines)
        {
            if (line.Trim().StartsWith("min_asks"))
            {
                string minAsksString = line.Split('=')[1].Trim();
                if (int.TryParse(minAsksString, out int minAsks))
                {
                    MinimumAsks.Add(minAsks);
                }
            }
        }
    }

    // Update parameter bounds and reference values from the config file
    public void UpdateParameterBoundsAndReferenceValues()
    {
        string[] lines = configText.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);

        foreach (string line in lines)
        {
            if (line.StartsWith("lb"))
            {
                string lbValues = line.Split('=')[1].Trim();
                ParameterLowerBounds = lbValues.Trim('[', ']').Split(',').Select(float.Parse).ToList();
            }
            else if (line.StartsWith("ub"))
            {
                string ubValues = line.Split('=')[1].Trim();
                ParameterUpperBounds = ubValues.Trim('[', ']').Split(',').Select(float.Parse).ToList();
            }
        }

        if (ParameterLowerBounds.Count >= 2 && ParameterUpperBounds.Count >= 2)
        {
            Wdim1Ref = ((ParameterUpperBounds[0] - ParameterLowerBounds[0]) / 2.0f + ParameterLowerBounds[0]);
            Wdim2Ref = ((ParameterUpperBounds[1] - ParameterLowerBounds[1]) / 2.0f + ParameterLowerBounds[1]);
        }
        else
        {
            Debug.LogError("Parameter bounds are not correctly set in the config file.");
        }
    }

    // Compute cumulative sum of minimum asks
    public void ComputeCumulativeSum()
    {
        CumulativeAsks.Clear();
        int runningTotal = 0;
        foreach (int value in MinimumAsks)
        {
            runningTotal += value;
            CumulativeAsks.Add(runningTotal);
        }
    }

    // Construct Sobol scaler list based on trials and shuffle a portion
    public void ConstructSobolScalerList(List<float> sobol_scaler, bool flag_noScale = false)
    {
        if (!flag_noScale)
        {
            SobolScalerList.Clear();

            // Repeat each sobol_scaler value based on the corresponding trials count
            for (int i = 0; i < sobol_scaler.Count; i++)
            {
                for (int j = 0; j < MinimumAsks[i]; j++)
                {
                    SobolScalerList.Add(sobol_scaler[i]);
                }
            }

            // Shuffle the Sobol scaler list for the required trials
            int optStratIndex = StrategyNames.IndexOf("opt_strat");
            if (optStratIndex >= 0 && optStratIndex < CumulativeAsks.Count)
            {
                int numShuffleTrials = CumulativeAsks[optStratIndex - 1];
                System.Random rng = new System.Random();

                SobolScalerList = SobolScalerList.Take(numShuffleTrials)
                    .OrderBy(x => rng.Next())
                    .Concat(SobolScalerList.Skip(numShuffleTrials))
                    .ToList();
            }
            else
            {
                Debug.LogError("Strategy 'opt_strat' not found or cumulative asks not computed.");
            }
        }
        else
        {
            // If MOCS is enabled, the sobol scaler is default to 1.0  (no scaling)
            // the number of elements matches the number of trials CumulativeAsks.Count
            SobolScalerList = Enumerable.Repeat(1.0f, CumulativeAsks[CumulativeAsks.Count - 1]).ToList();
        }
    }
}

// NOTE: This entire class is now obsolete. It was originally used in the pilot experiment 
// (interleaved 2D thresholding task), where both trial placement and color transformation 
// were handled in Unity.
// In that version, trial selection was performed in the Wishart space, and the selected 
// values were then transformed into RGB space for stimulus presentation. The transformation 
// matrix was computed using the monitor's spectral characteristics.
// In the updated experiment (4D thresholding task), this transformation is NO LONGER NEEDED 
// in Unity. Instead, Python and Unity communicate via a text file, where Python determines 
// the next trial and directly sends the pre-transformed RGB values. As a result, Unity no 
// longer needs to perform any Wishart-to-RGB conversion.
public class TransformationHelper
{
    private string transformationFile;  // Path to CSV file for W to RGB matrix
    private float[,] M1; // Matrix for W-space to RGB transformation

    public TransformationHelper(string transformationFile)
    {
        this.transformationFile = transformationFile;
        LoadTransformationMatrices();
    }

    // make M1 accessible outside the class
    public float[,] Mtransform => M1;

    // Method to load transformation matrices from CSV files
    private void LoadTransformationMatrices()
    {
        // Load M1 matrix (W to RGB)
        M1 = LoadMatrixFromFile(transformationFile);

        if (M1 == null)
        {
            Debug.LogError("Failed to load transformation matrices.");
        }
        else
        {
            Debug.Log("Transformation matrices loaded successfully.");
        }
    }

    // Method to convert W-space values to RGB values
    public List<float> WToRGB(float w1, float w2, float w3)
    {
        float r = M1[0, 0] * w1 + M1[0, 1] * w2 + M1[0, 2] * w3;
        float g = M1[1, 0] * w1 + M1[1, 1] * w2 + M1[1, 2] * w3;
        float b = M1[2, 0] * w1 + M1[2, 1] * w2 + M1[2, 2] * w3;

        return new List<float> { r, g, b };
    }

    // Helper method to load a 3x3 matrix from a CSV file
    private float[,] LoadMatrixFromFile(string filePath)
    {
        if (!File.Exists(filePath))
        {
            Debug.LogError($"File not found: {filePath}");
            return null;
        }

        try
        {
            string[] lines = File.ReadAllLines(filePath);
            float[,] matrix = new float[3, 3];

            for (int i = 0; i < 3; i++)
            {
                string[] values = lines[i].Split(',');

                for (int j = 0; j < 3; j++)
                {
                    matrix[i, j] = float.Parse(values[j]);
                }
            }

            return matrix;
        }
        catch (Exception e)
        {
            Debug.LogError($"Error loading matrix from file: {filePath}\n{e.Message}");
            return null;
        }
    }
}

// NOTE: This entire class is now obsolete. It was originally used in the pilot experiment 
// (interleaved 2D thresholding task) when the AEPsych C# client was in use. 
// The purpose of this class was to store additional trial data beyond just the 
// AEPsych-selected trial and participant response. It allowed us to append extra 
// information before saving the data to the database.
// In the updated experiment, this functionality is no longer needed, as we can just output
// the additional trial data to CSV files.
public class MyExperimentTrialInfo : TrialMetadata
{
    public int blockCounter;
    public int trialCounter;
    public int OddLocation;
    public float solbolScaler;
    public bool flag_MOCS;
    public List<float> W_ref;
    public List<float> W_comp;
    public List<float> RGB_ref;
    public List<float> RGB_comp;
    public List<float> RGB_ref_corrected;
    public List<float> RGB_comp_corrected;
    public List<bool> pressedKeys;
    // inverse gamma table
    public List<double> interpLuminance; 
    public List<double> redCorrected;    
    public List<double> greenCorrected;  
    public List<double> blueCorrected;   
    // Transformation matrix (M1) for W to RGB conversion
    public float[,] TransformationMatrix; 
    public Dictionary<string, float> durations;
    public string trialDateTime; // Nullable DateTime to ensure it's set only if required

    public MyExperimentTrialInfo(
        int blockCounter,
        int trialCounter,
        float responseTime,
        int OddLocation,
        float sobol_scaler,
        bool flag_MOCS,
        string participantId = "",
        List<float> W_ref = null,
        List<float> W_comp = null,
        List<float> RGB_ref = null,
        List<float> RGB_comp = null,
        List<float> RGB_ref_corrected = null,
        List<float> RGB_comp_corrected = null,
        List<bool> pressedKeys = null,
        List<double> interpLuminance = null,
        List<double> redCorrected = null,
        List<double> greenCorrected = null,
        List<double> blueCorrected = null,
        float[,] transformationMatrix = null,
        Dictionary<string, float> durations = null)
        : base(responseTime, participantId)
    {
        this.blockCounter = blockCounter;
        this.trialCounter = trialCounter;
        this.OddLocation = OddLocation;
        this.solbolScaler = sobol_scaler;
        this.flag_MOCS = flag_MOCS;
        this.W_ref = W_ref ?? new List<float> { 0, 0, 0 };
        this.W_comp = W_comp ?? new List<float> { 0, 0, 0 };
        this.RGB_ref = RGB_ref ?? new List<float> { 0, 0, 0 };
        this.RGB_comp = RGB_comp ?? new List<float> { 0, 0, 0 };
        this.RGB_ref_corrected = RGB_ref_corrected ?? new List<float> { 0, 0, 0 };
        this.RGB_comp_corrected = RGB_comp_corrected ?? new List<float> { 0, 0, 0 };

        this.pressedKeys = pressedKeys ?? new List<bool> { false, false, false };

        // Save gamma correction data only for the first trial
        this.interpLuminance = trialCounter == 1 ? interpLuminance : new List<double>();
        this.redCorrected = trialCounter == 1 ? redCorrected : new List<double>();
        this.greenCorrected = trialCounter == 1 ? greenCorrected : new List<double>();
        this.blueCorrected = trialCounter == 1 ? blueCorrected : new List<double>();

        // Save the transformation matrix only for the first trial
        this.TransformationMatrix = trialCounter == 1 ? transformationMatrix : new float[3, 3];

        // Save the duration values (fixation, blank, stimulus, feedback)
        this.durations = durations ?? new Dictionary<string, float>();

        // Save the date and time as a formatted string, or an empty string if not the first trial
        this.trialDateTime = trialCounter == 1 ? DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") : "";
    }
}

