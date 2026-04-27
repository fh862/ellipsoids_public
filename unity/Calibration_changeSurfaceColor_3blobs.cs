using System.Collections;
using System.IO;
using UnityEngine;
using System;
using System.Text.RegularExpressions;

public class Calibration_changeSurfaceColor : MonoBehaviour
{
    // Start is called before the first frame update
    public Camera mainCamera;
    public GameObject topBlob;
    public GameObject leftBlob;
    public GameObject rightBlob;
    public GameObject topCube;
    public GameObject leftCube;
    public GameObject rightCube;
    public Transform topRoom;
    public string blobLocation; //which blobby stimulus we would like to modify color and do measurements

    public float WaitTime_measurements = 5; //how much time do we wait for each measurement
    public float waitTime_adjustPhotometer = 240; //how much time do we wait before we start the first measurement
    public bool gammaCorrection_blob = false; // Gamma correction for the color values
                                              // Note that if this is a Klein test, we should also set gammaCorrect to be true so that the luminance averaged
                                              // across the aperture of the Klein increases linearly
    public string gammaCorrection_blob_fileDate = "02012026";
    private string gammaCorrectionFile_blob;
    public bool gammaCorrection_background = false; //Gamma correction for the camera's background.
                                                    //We need different knobs for the blobby stimuli and for the camera's background
                                                    //as they way they are rendered are different from each other.
    public string gammaCorrection_background_fileDate = "02102026";
    private string gammaCorrectionFile_background;
    public bool KleinTest = false;
    public bool calibrateBackground = false;
    private bool removeBlob_initScreen = false;

    private bool changeSurfaceTexture = true; // Change the property of the texture OR the surface color
    private bool saveFrameBuffer => KleinTest;

    // Save the frame buffer as an EXR file, which is much faster compared to saving 8-bit RGB values to a txt file
    // if it's a Klein test, this variable should be true because the goal is to examine color depth by 
    // measuring luminance increments and compare those with exr file outputs
    // if it's not a Klein test, then we do not save the frame buffer as exr file

    private GameObject blob;
    private Renderer blobRenderer;
    private GameObject cube;
    private Renderer cubeRenderer;
    private bool isCalibrationDone = false;
    private string lastProcessedLine = "";
    private bool screenshotTaken = false; // To ensure only one screenshot (.png format) is taken
    private string initFilePath;
    private string saveDirectory;
    private bool save_8bitRGB_txt = false; //To save 8-bit RGB values to a txt file, which takes quite long time

    private Color initColor = new Color(0.2404f, 0.2975f, 0.4527f); //new Color(0.2390f, 0.3086f, 0.4876f); 
    private Color initCameraBackground = new Color(0.6045f, 0.6166f, 0.6402f);//new Color(0.4741f, 0.6123f, 0.9673f);
    private float smoothness = 0.0f;
    // Initialize a counter 
    private int frameCounter = 1;
    // classes
    private GammaCorrectionHelper gammaCorrectionHelper_blob;
    private GammaCorrectionHelper gammaCorrectionHelper_background;
    private SurfaceColorChanger surfaceColorChanger;
    private CameraBackgroundChanger cameraBackgroundChanger;
    private FileDataExtractor fileDataExtractor;

    void Start()
    {
        // Define the save directory based on the current date
        string saveDir = $"D:/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_materials/Calibration/{DateTime.Now.ToString("MMddyyyy")}/";
        saveDirectory = saveDir;

        // Check if the save directory exists; if not, throw an error
        if (!Directory.Exists(saveDir))
        {
            Debug.LogError($"❌ Error: The directory '{saveDir}' does not exist. Check file path.");
            return; // Stop further execution safely
        }

        // Define the path to the initialization log file (from MATLAB to Unity)
        initFilePath = Path.Combine(saveDirectory, $"MATLAB_Unity_calibration_log_{DateTime.Now.ToString("MMddyyyy")}.txt");

        // Check if the initialization file exists; if not, throw an error
        if (!File.Exists(initFilePath))
        {
            Debug.LogError($"❌ Error: The file '{initFilePath}' does not exist. Ensure MATLAB writes it.");
            return; // Stop execution safely
        }

        // Load gamma correction data for the blobby stimuli if enabled
        if (gammaCorrection_blob)
        {
            gammaCorrectionFile_blob =
                $"D:/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_materials/Calibration/"+
                $"Inverse_gamma_functions_{gammaCorrection_blob_fileDate}/" +
                $"DELL_{gammaCorrection_blob_fileDate}_texture_right.csv";           
            
            // Check if the gamma correction file exists; if not, throw an error
            if (!File.Exists(gammaCorrectionFile_blob))
            {
                Debug.LogError($"❌ Error: The file '{gammaCorrectionFile_blob}' does not exist.");
                return; // Stop execution safely
            }
            gammaCorrectionHelper_blob = new GammaCorrectionHelper(gammaCorrectionFile_blob);
            gammaCorrectionHelper_blob.LoadGammaCorrectionData();
        }

        // Initialize the surface color changer utility with gamma correction and smoothness settings
        surfaceColorChanger = new SurfaceColorChanger(gammaCorrectionHelper_blob, gammaCorrection_blob, smoothness);

        // Load gamma correction data for the camera's background if enabled
        if (gammaCorrection_background)
        {
            gammaCorrectionFile_background =
                $"D:/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_materials/Calibration/" +
                $"Inverse_gamma_functions_{gammaCorrection_background_fileDate}/" +
                $"DELL_{gammaCorrection_background_fileDate}_background.csv";   

            // Check if the gamma correction file exists; if not, throw an error
            if (!File.Exists(gammaCorrectionFile_background))
            {
                Debug.LogError($"❌ Error: The file '{gammaCorrectionFile_background}' does not exist.");
                return; // Stop execution safely
            }
            gammaCorrectionHelper_background = new GammaCorrectionHelper(gammaCorrectionFile_background);
            gammaCorrectionHelper_background.LoadGammaCorrectionData();            
        }

        cameraBackgroundChanger = new CameraBackgroundChanger(gammaCorrectionHelper_background, gammaCorrection_background);

        // check if we are using Klein 10A to measure luminance. If so, update some settings
        ApplyKleinTestSettings();

        // Select the appropriate blob and renderer based on blobLocation (e.g., "top", "left", "right")
        SelectBlobAndRenderer();

        // Initialize the file data extractor for reading from the calibration log
        fileDataExtractor = new FileDataExtractor();

        // Start the coroutine to check the initialization file and set up the screen
        StartCoroutine(Check_initializationFile_startSettingup());
    }

    void Update()
    {
        if (!isCalibrationDone)
        {
            // Read the last line of the file
            string lastLine = fileDataExtractor.ReadLastLine(initFilePath);

            // Only process if the line has changed
            if (lastLine != lastProcessedLine)
            {
                lastProcessedLine = lastLine;

                var (isDone, _) = fileDataExtractor.CheckLastNthCharacters(lastLine, "Done", 0);
                var (isDisplay, _) = fileDataExtractor.CheckLastNthCharacters(lastLine, "Image_Display", 0);

                if (isDone)
                {
                    isCalibrationDone = true;
                    // End the game mode in Unity Editor
                    UnityEditor.EditorApplication.isPlaying = false;
                }
                else if (isDisplay)
                {
                    // Start capturing frames only if we are not already capturing
                    if (!calibrateBackground)
                    {
                        // Calibrate the blobby stimuli
                        StartCoroutine(ChangeSurfaceRepeatedly());
                    }
                    else
                    {
                        // Calibrate the camera background
                        StartCoroutine(ChangeBackgroundRepeatedly());
                    }
                }
            }
        }
    }

    /// <summary>
    /// set the camera's background to display solid color
    /// Configures the scene based on whether KleinTest is enabled.
    /// 
    /// If this is a Klein test:
    /// - The blob location is forced to "top".
    /// - The other two blobs and their cubes are disabled.
    /// - The camera background is set to white to facilitate easy stimulus extraction from EXR files
    ///   (by identifying all pixels with a value of 1 and excluding them).
    /// - The camera is repositioned to align with the center of the `topRoom` to properly frame the stimulus.
    ///
    /// If this is not a Klein test:
    /// - The camera background is set to an achromatic color (`initCameraBackground`).
    /// - The camera remains in its original position (no repositioning).
    private void ApplyKleinTestSettings()
    {
        if (mainCamera != null && mainCamera.clearFlags != CameraClearFlags.SolidColor)
            mainCamera.clearFlags = CameraClearFlags.SolidColor;

        if (KleinTest)
        {
            //if it's a Klein test, we cannot also be calibrating the background
            if (calibrateBackground)
            {
                const string err = "KleinTest is ON, so calibrateBackground must be false.";
                Debug.LogError(err);                 // optional, for the console
                throw new InvalidOperationException(err);
            }

            // 🔝 Force blob location to "top" since only the top blob is measured in KleinTest mode
            blobLocation = "top";

            // 🚫 Disable the left and right blobs along with their cubes
            leftBlob.SetActive(false);
            leftCube.SetActive(false);
            rightBlob.SetActive(false);
            rightCube.SetActive(false);

            // Set camera background to **white** for easier stimulus extraction from EXR files
            mainCamera.backgroundColor = Color.white;

            // Move the camera to **align with the center of the parent object (`topRoom`)**
            if (topRoom != null)
            {
                Vector3 parentPosition = topRoom.transform.position;
                Debug.Log($"topRoom Position - X: {parentPosition.x}, Y: {parentPosition.y}, Z: {parentPosition.z}");

                // Set camera position to align with `topRoom`, keeping Z at 0 for correct viewing
                mainCamera.transform.position = new Vector3(parentPosition.x, parentPosition.y, 0);
            }
            else
            {
                Debug.LogError("❌ `topRoom` is not assigned! Camera position not updated.");
            }
        }
        else
        {
            // If not a Klein test, set the background to an achromatic color (default initialization)
            cameraBackgroundChanger.ChangeCameraBackground(mainCamera, initCameraBackground.r, initCameraBackground.g, initCameraBackground.b);
        }
    }

    /// <summary>
    /// Selects which blob to calibrate based on the specified `blobLocation`.
    /// There are three blobs: top, bottom left, and bottom right. 
    /// - The selected blob and its corresponding cube will be assigned for calibration.
    /// - If `centerOneBlob` is enabled and the top blob is selected, the scene adjusts to focus only on that blob.
    /// - The function also ensures that the selected objects use the Standard shader.
    private void SelectBlobAndRenderer()
    {
        // 🔝 Determine which blob and renderer to use based on the selected blob location
        if (blobLocation.ToLower() == "top")
        {
            // Select the top blob and its corresponding cube
            blob = topBlob;
            cube = topCube;
            blobRenderer = topBlob.GetComponent<Renderer>();
            cubeRenderer = topCube.GetComponent<Renderer>();

            // 🚫 If we want to center only this blob for calibration, adjust the scene accordingly
            if (KleinTest)
            {
                // Disable the cube, so it'd be easier later to extract the stimulus from .exr files
                cube.SetActive(false);
            }
        }
        else if (blobLocation.ToLower() == "left")
        {
            // 👈 Select the left blob and its corresponding cube
            blob = leftBlob;
            cube = leftCube;
            blobRenderer = leftBlob.GetComponent<Renderer>();
            cubeRenderer = leftCube.GetComponent<Renderer>();
        }
        else if (blobLocation.ToLower() == "right")
        {
            // 👉 Select the right blob and its corresponding cube
            blob = rightBlob;
            cube = rightCube;
            blobRenderer = rightBlob.GetComponent<Renderer>();
            cubeRenderer = rightCube.GetComponent<Renderer>();
        }
        else
        {
            throw new ArgumentException("Invalid blob location specified.");
        }

        // Ensure the selected blob and cube use the Standard shader for consistent rendering
        blobRenderer.material.shader = Shader.Find("Standard");
        cubeRenderer.material.shader = Shader.Find("Standard");
    }

    /// <summary>
    /// Waits for the initialization file to be created and verifies that it contains 
    /// the 'Initialize_Unity' command. Once verified, it initializes the stimuli and 
    /// writes back to the text file to confirm that initialization is complete.
    private IEnumerator Check_initializationFile_startSettingup()
    {
        // ✅ Wait for the initialization file and verify its contents
        string command_start = "Initialize_Unity";
        yield return StartCoroutine(fileDataExtractor.VerifyFileExistsAndCheckNthLastCharacters(initFilePath, command_start, 0));
        Debug.Log("Initialization file found and verified.");

        // ✅ Set up the screen and initialize the stimuli
        Debug.Log("Starting screen setup...");
        yield return StartCoroutine(SetupScreen());
        Debug.Log("Screen setup completed.");

        // ✅ Write a message to the text file confirming setup completion
        fileDataExtractor.WriteMessageToFile(initFilePath, "Finished_Screen_Setup");
        Debug.Log("'Finished_Screen_Setup' message written to the file.");
    }

    /// <summary>
    /// Initializes the surface color of the blob and the cube, and then pauses 
    /// for a specified duration (`waitTime_adjustRadiometer`) to allow the experimenter 
    /// to align the radiometer with the stimulus before proceeding.
    /// </summary>
    public IEnumerator SetupScreen()
    {
        // 🎨 Initialize surface colors for blob and cube
        if (changeSurfaceTexture)
        {
            if (!KleinTest && cubeRenderer != null && cube.activeSelf)
            {
                surfaceColorChanger.ChangeSurfaceTextureColor(cubeRenderer, initColor.r, initColor.g, initColor.b);
            }

            if (!removeBlob_initScreen)
            {
                surfaceColorChanger.ChangeSurfaceTextureColor(blobRenderer, initColor.r, initColor.g, initColor.b);
            }
            else
            {
                // 🚫 Disable the left and right blobs along with their cubes
                topBlob.SetActive(false);
                leftBlob.SetActive(false);
                rightBlob.SetActive(false);
            }
        }
        else
        {
            if (!KleinTest && cubeRenderer != null && cube.activeSelf)
            {
                surfaceColorChanger.ChangeSurfaceColor(cubeRenderer, initColor.r, initColor.g, initColor.b);
            }

            if (!removeBlob_initScreen)
            {
                surfaceColorChanger.ChangeSurfaceColor(blobRenderer, initColor.r, initColor.g, initColor.b);
            }
            else
            {
                // 🚫 Disable the left and right blobs along with their cubes
                topBlob.SetActive(false);
                leftBlob.SetActive(false);
                rightBlob.SetActive(false);
            }
        }

        // ⏳ Pause to allow the experimenter to align the radiometer with the stimulus
        Debug.Log($"Pausing for {waitTime_adjustPhotometer} seconds to align radiometer...");
        yield return new WaitForSeconds(waitTime_adjustPhotometer);
        Debug.Log("Pause complete, proceeding with the experiment...");
    }

    /// <summary>
    /// This method repeatedly updates the surface color of the cube and the blobby stimulus 
    /// based on RGB values extracted from a text file. It supports both texture-based and 
    /// direct color changes. If performing a Klein test, only the blobby stimulus is modified.
    /// 
    /// Additional functionalities include:
    /// - Capturing the rendered frame buffer as an .exr file (if enabled).
    /// - Taking a screenshot after the first frame update.
    /// - Writing a confirmation message to the text file.
    /// - Waiting for a specified duration before the next iteration.
    private IEnumerator ChangeSurfaceRepeatedly()
    {
        // 📄 Read the last line from the text file to extract color information
        string colorString_line = fileDataExtractor.ReadLastLine(initFilePath);
        string[] parts = colorString_line.Split(':');
        string colorString = parts[^1].Trim();
        string[] colorStrings = colorString.Split(',');

        // 🎨 Extract background and target stimulus color values
        string bgColorString = colorStrings[0].Replace("Background_Display", "").Trim();  // Background color
        string sColorString = colorStrings[1].Replace("Image_Display", "").Trim();  // Target stimulus color

        Debug.Log($"Background color string: {bgColorString}");
        Debug.Log($"Target color string: {sColorString}");

        // 🎨 Extract RGB values for the background and stimulus
        var (bgR, bgG, bgB) = fileDataExtractor.ExtractRGBValues(bgColorString);
        var (r, g, b) = fileDataExtractor.ExtractRGBValues(sColorString);

        // 🖌️ Apply colors to surfaces (either via texture or direct surface color change)
        if (changeSurfaceTexture)
        {
            if (!KleinTest)
            {
                // Update cube color if not in Klein test mode
                surfaceColorChanger.ChangeSurfaceTextureColor(cubeRenderer, bgR, bgG, bgB);
            }
            // Always update the blobby stimulus
            surfaceColorChanger.ChangeSurfaceTextureColor(blobRenderer, r, g, b);
        }
        else
        {
            if (!KleinTest)
            {
                // Update cube color if not in Klein test mode
                surfaceColorChanger.ChangeSurfaceColor(cubeRenderer, bgR, bgG, bgB);
            }
            // Always update the blobby stimulus
            surfaceColorChanger.ChangeSurfaceColor(blobRenderer, r, g, b);
        }

        // 🕒 Wait for the frame to finish rendering before proceeding
        yield return new WaitForEndOfFrame();

        // 📸 Capture the frame buffer and save as an .exr file (if enabled)
        if (saveFrameBuffer)
        {
            FrameBufferUtils.CaptureRenderFrameBuffer(mainCamera, saveDirectory, frameCounter);
            yield return new WaitForEndOfFrame(); // Ensure capture is complete
        }

        // 📷 Take a screenshot after the first frame update (only once per iteration)
        if (!screenshotTaken)
        {
            string filename = Path.Combine(saveDirectory, $"Background_R{bgR}_G{bgG}_B{bgB}_Target_R{r}_G{g}_B{b}.png");
            ScreenCapture.CaptureScreenshot(filename);
            screenshotTaken = true;  // Prevent multiple screenshots in the same iteration
            Debug.Log($"📷 Screenshot saved: {filename}");
        }
        yield return new WaitForEndOfFrame();  // Ensure screenshot is fully processed
        screenshotTaken = false;  // Reset flag for next iteration

        // ✅ Write a confirmation message to the text file indicating the image was displayed
        fileDataExtractor.WriteMessageToFile(initFilePath, $"{colorString} Image_Successfully_Displayed");

        // ⏳ Pause for a specified duration (SOA) before the next update
        yield return new WaitForSeconds(WaitTime_measurements);

        // 🔄 Increment frame counter for the next iteration
        frameCounter += 1;
    }

    /// <summary>
    /// Variant of ChangeSurfaceRepeatedly() that:
    /// 1) Reads the last line of the init file to get two colors:
    ///    - "Background_Display" → applied to the cubic room + blob
    ///    - "Image_Display"      → applied to the **camera background**
    /// 2) Updates scene colors
    /// 3) Optionally grabs a frame buffer and a screenshot
    /// 4) Logs success back to the init file and waits SOA
    /// </summary>
    private IEnumerator ChangeBackgroundRepeatedly()
    {
        // 📄 Read the last line from the text file to extract color information
        string colorString_line = fileDataExtractor.ReadLastLine(initFilePath);
        string[] parts = colorString_line.Split(':');
        string colorString = parts[^1].Trim();
        string[] colorStrings = colorString.Split(',');

        // 🎨 Extract background and target stimulus color values
        string bgColorString = colorStrings[0].Replace("Background_Display", "").Trim();  // Background color
        string sColorString = colorStrings[1].Replace("Image_Display", "").Trim();  // Target stimulus color

        Debug.Log($"Background color string: {bgColorString}");
        Debug.Log($"Target color string: {sColorString}");

        // 🎨 Extract RGB values for the background and stimulus
        var (bgR, bgG, bgB) = fileDataExtractor.ExtractRGBValues(bgColorString);
        var (r, g, b) = fileDataExtractor.ExtractRGBValues(sColorString);

        // --- CAMERA BACKGROUND (Image_Display) ---
        cameraBackgroundChanger.ChangeCameraBackground(mainCamera, r, g, b);

        // --- ROOM + BLOB (Background_Display) ---
        surfaceColorChanger.ChangeSurfaceColor(cubeRenderer, bgR, bgG, bgB);
        surfaceColorChanger.ChangeSurfaceColor(blobRenderer, bgR, bgG, bgB);

        // 🕒 Wait for the frame to finish rendering before proceeding
        yield return new WaitForEndOfFrame();

        // 📸 Capture the frame buffer and save as an .exr file (if enabled)
        if (saveFrameBuffer)
        {
            FrameBufferUtils.CaptureRenderFrameBuffer(mainCamera, saveDirectory, frameCounter);
            yield return new WaitForEndOfFrame(); // Ensure capture is complete
        }

        // 📷 Take a screenshot after the first frame update (only once per iteration)
        if (!screenshotTaken)
        {
            string filename = Path.Combine(saveDirectory, $"Background_R{bgR}_G{bgG}_B{bgB}_Target_R{r}_G{g}_B{b}.png");
            ScreenCapture.CaptureScreenshot(filename);
            screenshotTaken = true;  // Prevent multiple screenshots in the same iteration
            Debug.Log($"📷 Screenshot saved: {filename}");
        }
        yield return new WaitForEndOfFrame();  // Ensure screenshot is fully processed
        screenshotTaken = false;  // Reset flag for next iteration

        // ✅ Write a confirmation message to the text file indicating the image was displayed
        fileDataExtractor.WriteMessageToFile(initFilePath, $"{colorString} Image_Successfully_Displayed");

        // ⏳ Pause for a specified duration (SOA) before the next update
        yield return new WaitForSeconds(WaitTime_measurements);

        // 🔄 Increment frame counter for the next iteration
        frameCounter += 1;
    }
}
