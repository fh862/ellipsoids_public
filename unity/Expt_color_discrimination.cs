using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using UnityEngine;
using TMPro;
using System;
using UnityEditor;
using UnityEngine.InputSystem;

public class Expt_color_discrimination : MonoBehaviour
{
    // **Experiment Objects and Components**
    // Do not modify these object assignments
    public GameObject TopObj;
    public Renderer RendererTopObj;
    public GameObject LeftObj;
    public Renderer RendererLeftObj;
    public GameObject RightObj;
    public Renderer RendererRightObj;
    public GameObject TopObj_room;
    public Renderer RendererTopObj_room;
    public GameObject LeftObj_room;
    public Renderer RendererLeftObj_room;
    public GameObject RightObj_room;
    public Renderer RendererRightObj_room;

    public AudioSource audioSource;
    public AudioClip correctSound;
    public AudioClip incorrectSound;    
    
    // UI elements for displaying instructions and feedback
    public TextMeshProUGUI displayText;

    // User response keys
    public KeyCode ResponseKeyLeft; 
    public KeyCode ResponseKeyRight;
    public KeyCode ResponseKeyTop;
    public KeyCode ResumeKey;  

    // **Experiment Parameters**
    public int subject_id;          // Subject ID
    public string subject_init;     // Subject initial
    public int session_today;       // session number (starting from 1)
    public bool is_practice = true; // Default to practice mode
    // This script is a general-purpose experiment controller that can run in different
    // stimulus spaces:
    //   (1) 2D isoluminant plane
    //   (2) 2D L–S isolating plane
    //   (3) full 3D gamut space
    public enum StimSpace { Isoluminant2D, LSIsolating2D, FullGamut3D, PartialGamut3D }

    [SerializeField]
    protected StimSpace stimSpace = StimSpace.FullGamut3D;

    // **Configuration and Calibration**
    /// <Note>: below, only three variables are declared as private —
    /// (1) gammaCorrection_fileDate, (2) gammaCorrectionFile, and (3) durations.
    /// All other non-public variables are marked as protected.
    /// 
    /// - Protected variables are inherited by subclasses and can be accessed or modified directly in those subclasses.
    /// - Private variables are only accessible within this base class; they cannot be read or written in a subclass.
    /// 
    /// These three variables are private because their values differ between the 4D oddity task
    /// and the suprathreshold task, so each subclass should define and manage its own versions.
    protected Gamepad gamepad;
    protected bool gammaCorrection = true; // Enable/disable gamma correction
    private string gammaCorrectionFile;
    protected float smoothness = 0.0f; // Default surface smoothness
    protected readonly List<float> room_colors = new List<float> { 0.3f, 0.3f, 0.3f }; // Background color of the scene
    protected int subject_seed; // Random seed for trial sequence
    protected int trialCounter = 0; // Tracks the trial number
    protected bool isExptDone = false; // Flag to indicate if the experiment is completed
    protected float codedAnswer; // Stores the participant's response
    protected float timeStamp_preStim_BlankScreen;
    protected float actualWait_preStim;
    protected float responseTime;
    protected string pressedKeysString;
    protected bool isRunningStimulus = false; 
    protected bool takeScreenshotOfStimuli = false; // only turn it on when we want to save some images for making figures
    protected bool screenshotTaken = false; // To ensure only one screenshot (.png format) is taken
    protected string lastImageDisplayCommand = "";

    // **Trial Timing Durations (in seconds)**
    private readonly Dictionary<string, float> durations = new Dictionary<string, float>
    {
        { "fixation", 0.5f },
        { "blank", 0.2f },
        { "stimulus", 1.0f },
        { "feedback", 0.5f },
        { "ITI", 1.5f},
        { "expected_RT", 0.2f} // Expected response time
    };

    // Map the selected stimulus space to the data subfolder name.
    // This is a read-only computed property, so it always stays consistent with stimSpace.
    // (You cannot accidentally set a mismatched folder name somewhere else.)
    private string SubfolderName => stimSpace switch
    {
        // Full 3D gamut experiment data go here
        StimSpace.FullGamut3D     => "6D_Expt_data",

        // L–S isolating 2D experiment (dichromat-related) data go here
        StimSpace.LSIsolating2D   => "4D_Expt_dichromats",

        // Measure the threshold around a fixed ref (refs) for dichromat subjects
        StimSpace.PartialGamut3D  => "3D_Expt_dichromats",

        // Isoluminant 2D experiment data (eLife dataset) go here
        StimSpace.Isoluminant2D   => "4D_Expt_data_eLife",

        // Safety net: enums are ints under the hood, so an invalid value is possible
        // (e.g., enum reordering/removal + old serialized scenes/prefabs, or manual casts).
        // Throwing makes the bug obvious instead of silently writing to a wrong folder.
        _ => throw new ArgumentOutOfRangeException(
            nameof(stimSpace),
            stimSpace,
            "Unknown StimSpace value"
        )
    };

    // Gamma correction LUT "date tag" depends on the stimulus space.
    // Keeping this as a computed property prevents mismatches.
    private string gammaCorrection_fileDate => stimSpace switch
    {
        StimSpace.Isoluminant2D   => "02242025", // eLife paper
        StimSpace.FullGamut3D     => "02012026", // 3D experiment, pilot: 07292025
        StimSpace.LSIsolating2D   => "10062025", // dichromat experiment
        StimSpace.PartialGamut3D  => "10062025", // same gamma correction used for the dichromat expt
        _ => throw new ArgumentOutOfRangeException(nameof(stimSpace), stimSpace, "Unknown StimSpace value")
    };

    // **File and Data Management**
    protected string saveDirectory;
    protected string filePrefix;
    protected string initFilePath;
    protected string csvFilePath;

    // **Helper Classes**
    protected GammaCorrectionHelper gammaCorrectionHelper;
    protected SurfaceColorChanger surfaceColorChanger;
    protected FileDataExtractor fileDataExtractor;
    protected TrialHelper trialHelper;

    /// <summary>
    /// Unity's Start() method, which runs once when the script is first executed.
    /// This method initializes the experiment by setting up subject information, file paths, 
    /// data storage, gamma correction, and helper classes. It also starts the screen setup process.
    void Start()
    {
        try
            {
            // 1️⃣🌱: Set the random seed for trial generation
            // The seed is used to ensure reproducibility of trials, keeping it consistent with Python scripts.
            // The formula varies based on whether it's a practice session:
            // - If it's a practice session → Seed = subject_id * 1000 + session_today
            // - Otherwise → Seed = subject_id * 100 + session_today
            subject_seed = is_practice ? subject_id * 10000 + session_today : subject_id * 100 + session_today;

            // 2️⃣📁: Determine file paths for saving experiment data
            // This sets the correct save directory and initialization file path based on subject ID, session number, and whether it's a practice session.
            DetermineFilePaths(subject_id, subject_init, session_today, is_practice, SubfolderName);

            // 3️⃣🎨: Load gamma correction data if enabled
            // If gamma correction is enabled, it loads the correction data from the specified file.
            if (gammaCorrection)
            {
                gammaCorrectionFile = $"D:/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_materials/Calibration/"+
                $"Inverse_gamma_functions_{gammaCorrection_fileDate}/"+
                $"DELL_{gammaCorrection_fileDate}_texture_right.csv";

                // Check if the gamma correction file exists; if not, throw an error
                if (!File.Exists(gammaCorrectionFile))
                {
                    Debug.LogError($"❌ Error: The file '{gammaCorrectionFile}' does not exist.");
                    return; // Stop execution safely
                }
                gammaCorrectionHelper = new GammaCorrectionHelper(gammaCorrectionFile);
                gammaCorrectionHelper.LoadGammaCorrectionData();
            }

            // 4️⃣📰: Create a CSV file for recording trial data
            // This ensures that each session has its own log file to store responses and trial information.
            createTrialData();

            // 5️⃣🎮: connet to the game pad
            ConnectGamepads();

            // 6️⃣🤝: Initialize helper classes
            // These helper classes assist with color manipulation, file handling, and trial management.
            surfaceColorChanger = new SurfaceColorChanger(gammaCorrectionHelper, gammaCorrection, smoothness); // Handles color adjustments
            fileDataExtractor = new FileDataExtractor(); // Handles reading and writing experiment data
            trialHelper = new TrialHelper(subject_seed); // Generates randomized trial sequences based on subject seed

            // 7️⃣🔁: Start screen setup and experiment initialization
            // This coroutine ensures that the initialization file is correctly set up before starting the experiment.
            StartCoroutine(Check_initializationFile_startSettingup());
        }
        catch (Exception e)
        {
            Debug.LogError($"Error in Update: {e.Message}");
            UnityEditor.EditorApplication.isPlaying = false; // Stop execution
        }
    }

    /// <summary>
    /// Unity's Update() method, which runs once per frame.
    /// This function continuously checks the last line of an external file to determine the current experiment state.
    /// Based on the detected state, it initiates the appropriate coroutine (trial execution, break handling, or experiment termination).

    void Update() // 🔁 Runs every frame
    {
        try
        {
            if (!isExptDone && !isRunningStimulus) // ✅ Only process if no stimulus is running
            {
                // Read the most recent line from the experiment initialization file
                string lastLine = fileDataExtractor.ReadLastLine(initFilePath);
                Debug.Log($"Last read line: {lastLine}"); // ✅ Log the last read line

                // Check if the last line contains specific trigger words
                var (isDone, _) = fileDataExtractor.CheckLastNthCharacters(lastLine, "Done", 0);
                var (isDisplay, _) = fileDataExtractor.CheckLastNthCharacters(lastLine, "Image_Display", 0);
                var (isBreak, _) = fileDataExtractor.CheckLastNthCharacters(lastLine, "Break", 0);

                // Ensure we are not already running a stimulus (prevents duplicate trials)
                if (isDone) // 🔚 If "Done" is detected, terminate the experiment
                {
                    isExptDone = true; // Flag the experiment as completed
                    StartCoroutine(CloseScreen()); // Start the closing procedure
                }
                else if (isDisplay) // 🔛 If "Image_Display" is detected, check for duplicates
                {
                    if (lastLine == lastImageDisplayCommand) // 🚨 Prevent duplicate trials
                    {
                        Debug.Log("⚠ Duplicate 'Image_Display' command detected. Skipping trial.");
                        return; // Exit early to avoid re-running the same trial
                    }

                    // ✅ Store the new Image_Display command
                    lastImageDisplayCommand = lastLine;

                    isRunningStimulus = true; // Prevent overlapping execution
                    Debug.Log("Starting HandleStimulusDisplay() coroutine.");
                    StartCoroutine(HandleStimulusDisplay(lastLine)); // Process and display the stimulus for a new trial
                }
                else if (isBreak) // ⏸ If "Break" is detected, enter the break phase
                {
                    isRunningStimulus = true;
                    StartCoroutine(ResumeExperiment()); // Wait for the participant to resume the experiment
                }
                else
                {
                    Debug.Log("No recognized command found. Waiting for next frame...");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error in Update: {e.Message}");
            UnityEditor.EditorApplication.isPlaying = false; // Stop execution
        }
    }

    /// <summary>
    /// Waits for the initialization file to be created, then sets up the screen.
    /// </summary>
    public IEnumerator Check_initializationFile_startSettingup()
    {
        // Wait until the initialization file is created
        string command_start = "Set_Up_to_Communicate";
        // This is a while loop. The default max wait time is 300 s to avoid indefinite loop
        yield return StartCoroutine(fileDataExtractor.VerifyFileExistsAndCheckNthLastCharacters(initFilePath, command_start, 0));

        // Once the file is found, set up the screen and start the calibration process
        Debug.Log("Initialization file found and verified. Starting screen setup.");
        InitializeScreen();
        Debug.Log("Screen setup completed.");

        // Write a message to the txt file to indicate the completion of the setup
        fileDataExtractor.WriteMessageToFile(initFilePath, "Ready_To_Communicate");
    }

    public void DetermineFilePaths(int subject_id, string subject_init, int session_today, bool is_practice, string subfolder_name = null)
    {
        // Determine save directory
        if (string.IsNullOrEmpty(subfolder_name))
        {
            // original behavior
            saveDirectory = is_practice
                ? $"B:\\sub{subject_id}\\practice\\"
                : $"B:\\sub{subject_id}\\";
        }
        else
        {
            // with subfolder
            saveDirectory = is_practice
                ? $"B:\\{subfolder_name}\\sub{subject_id}\\practice\\"
                : $"B:\\{subfolder_name}\\sub{subject_id}\\";
        }

        // Determine file prefix
        filePrefix = is_practice
            ? $"sub{subject_id}_{subject_init}_practice_session{session_today}_"
            : $"sub{subject_id}_{subject_init}_session{session_today}_";

        // Check if the directory exists
        Debug.Log($"🧪 Checking if saveDirectory exists: {saveDirectory}");
        if (!Directory.Exists(saveDirectory))
        {
            displayText.text = "Cannot find the directory!";
            throw new DirectoryNotFoundException($"The directory '{saveDirectory}' does not exist.");
        }
        Debug.Log($"Save directory set: {saveDirectory}");

        // Path to the initialization file
        initFilePath = GetMostRecentFile(saveDirectory, filePrefix);
        if (string.IsNullOrEmpty(initFilePath))
        {
            displayText.text = "Cannot find the file!";
            throw new FileNotFoundException($"No files found in {saveDirectory} starting with {filePrefix}");
        }
        Debug.Log($"Initialization file path set: {initFilePath}");
    }

    public string GetMostRecentFile(string directory, string filePrefix)
    {
        var matchingFiles = Directory.GetFiles(directory, $"{filePrefix}*")
                                    .OrderByDescending(f => File.GetLastWriteTime(f))
                                    .ToList();

        if (matchingFiles.Count > 0)
        {
            Debug.Log($"Most recent file found: {matchingFiles[0]}");
            return matchingFiles[0]; 
        }

        Debug.LogWarning($"No files found in {directory} starting with {filePrefix}");
        return string.Empty; // Return an empty string instead of null
    }

    public void InitializeScreen()
    {
        // Do not show the objects at the beginning
        RendererTopObj.enabled = false;
        RendererLeftObj.enabled = false;
        RendererRightObj.enabled = false;

        // Set room color
        surfaceColorChanger.ChangeSurfaceTextureColor(RendererTopObj_room, room_colors[0], room_colors[1], room_colors[2]);
        surfaceColorChanger.ChangeSurfaceTextureColor(RendererLeftObj_room, room_colors[0], room_colors[1], room_colors[2]);
        surfaceColorChanger.ChangeSurfaceTextureColor(RendererRightObj_room, room_colors[0], room_colors[1], room_colors[2]);
    }

    /// <summary>
    /// Gracefully closes the experiment by displaying "Done" for 5 seconds before stopping execution.
    public IEnumerator CloseScreen()
    {
        isRunningStimulus = true;
        displayText.text = "Done";
        yield return new WaitForSeconds(5.0f);
        // End the game mode in Unity Editor
        UnityEditor.EditorApplication.isPlaying = false;
        isRunningStimulus = false; 
    }

    /// <summary>
    /// Handles the break time between trials, allowing the user to resume when ready.
    public IEnumerator ResumeExperiment()
    {
        displayText.text = "Break time";
        // Wait until either the resume key or gamepad buttonSouth is pressed
        yield return new WaitUntil(() =>
        {
            if (Input.GetKeyDown(ResumeKey)) return true;
            if (gamepad != null && gamepad.buttonSouth.wasPressedThisFrame) return true;
            return false;
        });

        // Write 'resume' in the txt file so that the Python side knows
        fileDataExtractor.WriteMessageToFile(initFilePath, "Resume");

        displayText.text = "Resume";
        yield return new WaitForSeconds(1.0f);

        yield return StartCoroutine(NON_StimulusPresentation_PreStim());

        isRunningStimulus = false; 
    }


    /// <summary>
    /// Handles the sequence of stimulus presentation, post-response feedback, and pre-stimulus setup.
    /// This ensures all necessary steps are completed before the next trial starts.
    /// lastLine: The last recorded line in the initialization file.
    private IEnumerator HandleStimulusDisplay(string lastLine)
    {
        try
        {
            if (trialCounter == 0)
            {        
                // Countdown from 3 to 1
                for (int i = 5; i > 0; i--)
                {
                    displayText.text = i.ToString();
                    yield return new WaitForSeconds(1.0f);
                }

                displayText.text = "Begin";
                yield return new WaitForSeconds(1.0f);

                // Step 0: Run pre-stimulus presentation and WAIT until it finishes
                yield return StartCoroutine(NON_StimulusPresentation_PreStim());
            }
            // Step 1: Run stimulus presentation and WAIT until it finishes
            yield return StartCoroutine(StimulusPresentation(lastLine));

            // Step 2: Run post-response presentation and WAIT until it finishes
            yield return StartCoroutine(NON_StimulusPresentation_postResp());

            // Step 3: Run pre-stimulus presentation and WAIT until it finishes
            yield return StartCoroutine(NON_StimulusPresentation_PreStim());
        }
        finally
        {
            isRunningStimulus = false; // Ensure the flag is always reset
        }
    }


    /// <summary>
    /// Handles all timing events **except** stimulus presentation and response waiting.
    /// 
    /// This method consists of:
    /// 1️⃣ **Post-response events** (not shown on the first trial):
    ///    - A blank screen (0.2s)
    ///    - Visual and auditory feedback (0.5s)
    ///    - Inter-Trial Interval (ITI), normally 1.5s, but adjusted based on response time.
    ///
    /// 2️⃣ **Pre-stimulus events** (before showing the next stimulus):
    ///    - A fixation cross (0.5s)
    ///    - A blank screen (0.2s)
    ///
    /// 🕒 These timing events, along with the **expected 0.2s response time**, define the **2.9s**
    /// window for AEPsych to compute and determine the next trial placement.

    /// <summary>
    /// Displays visual feedback (text) and plays auditory feedback (sound) **simultaneously**.
    /// feedbackText: The feedback text to display (e.g., "Correct" or "Incorrect").
    /// feedbackSound: The corresponding sound effect for correct/incorrect responses.
    private void Show_Visual_Auditory_Feedback(string feedbackText, AudioClip feedbackSound)
    {
        // Assign the sound clip
        audioSource.clip = feedbackSound;
        
        // Play the sound and show text at the same time
        audioSource.Play();
        displayText.text = feedbackText;
    }

    /// <summary>
    /// Handles **pre-stimulus events**, ensuring a controlled viewing sequence before stimulus presentation.
    /// - Displays a fixation cross ("+") for 0.5s.
    /// - Shows a brief blank screen (0.2s) before the stimulus appears.
    public IEnumerator NON_StimulusPresentation_PreStim()
    {
        // Display fixation cross
        displayText.text = "+";

        // 📷 Take a screenshot after the first frame update (only once per iteration)
        if (takeScreenshotOfStimuli && trialCounter == 1)
        {
            string filename = Path.Combine(saveDirectory, $"fixation.png");
            ScreenCapture.CaptureScreenshot(filename);
            yield return new WaitForEndOfFrame();  // Ensure screenshot is fully processed
        }
        yield return new WaitForSeconds(durations["fixation"]);

        // Clear text for brief blank screen
        displayText.text = "";

        // 🔹 🕒 **NOW** Record the time between the '+' and stimulus presentation
        // because there might be a slight delay caused by reading files
        yield return new WaitForEndOfFrame();
        timeStamp_preStim_BlankScreen = Time.time;  

        // 📷 Take a screenshot after the first frame update (only once per iteration)
        if (takeScreenshotOfStimuli && trialCounter == 1)
        {
            string filename = Path.Combine(saveDirectory, $"blankscreen.png");
            ScreenCapture.CaptureScreenshot(filename);
            yield return new WaitForEndOfFrame();  // Ensure screenshot is fully processed
        }
        yield return new WaitForSeconds(durations["blank"]);
    }

    /// <summary>
    /// Handles **post-response events**, including feedback and ITI (Inter-Trial Interval).
    /// - Shows a brief blank screen (0.2s).
    /// - Displays feedback (text & sound) for 0.5s.
    /// - Waits for the ITI period (adjusted if response time was shorter than expected).
    public IEnumerator NON_StimulusPresentation_postResp()
    {        
        // Brief blank screen before feedback
        displayText.text = "";  
        yield return new WaitForSeconds(durations["blank"]);

        // 🔊 Show feedback with synchronized text and sound
        Show_Visual_Auditory_Feedback(
            codedAnswer == 1 ? "Correct" : "Incorrect",
            codedAnswer == 1 ? correctSound : incorrectSound
        );

        // 📷 Take a screenshot after the first frame update (only once per iteration)
        if (takeScreenshotOfStimuli && trialCounter == 1)
        {
            string filename = Path.Combine(saveDirectory, $"feedback.png");
            ScreenCapture.CaptureScreenshot(filename);
            yield return new WaitForEndOfFrame();  // Ensure screenshot is fully processed
        }

        // Wait while feedback is visible
        yield return new WaitForSeconds(durations["feedback"]);

        // Clear feedback text
        displayText.text = "";

        // Wait for ITI
        yield return new WaitForSeconds(durations["ITI"]);
    }

    /// <summary>
    /// Handles the presentation of visual stimuli and records the participant's response.
    /// last_line: The last recorded trial data from the experiment file.
    private IEnumerator StimulusPresentation(string last_line)
    {
        // Compute actual wait time after '+' disappears by taking the delay of reading files into account
        yield return new WaitForEndOfFrame(); // Ensure frame update happens
        if (timeStamp_preStim_BlankScreen > 0)
        {
            actualWait_preStim = Time.time - timeStamp_preStim_BlankScreen;
        }
        else
        {
            actualWait_preStim = 0.0f;
        }

        // Determine the odd object's location using the trial sequence generator
        int OddLocation = trialHelper.RefCompRandomSequence();

        // 📄 Extract trial information and reference/comparison stimulus colors from the file
        var (TrialInfo, RefColorString, CompColorString) = fileDataExtractor.ExtractTrialRefComp(last_line);

        // 🎨 Extract RGB values for the reference and comparison stimuli
        var (ref_R, ref_G, ref_B) = fileDataExtractor.ExtractRGBValues(RefColorString);
        var (comp_R, comp_G, comp_B) = fileDataExtractor.ExtractRGBValues(CompColorString);

        // 👀 Make stimuli visible
        RendererTopObj.enabled   = true;
        RendererLeftObj.enabled  = true;
        RendererRightObj.enabled = true;

        // Initialize variables for gamma-corrected RGB values
        Color RGB_ref_corrected = new Color(0f, 0f, 0f); 
        Color RGB_comp_corrected = new Color(0f, 0f, 0f); 

        // 🔄 Display stimuli with gamma correction applied (based on OddLocation)
        if (OddLocation == 1)
        {
            RGB_comp_corrected = surfaceColorChanger.ChangeSurfaceTextureColor(RendererTopObj, comp_R, comp_G, comp_B);
            RGB_ref_corrected = surfaceColorChanger.ChangeSurfaceTextureColor(RendererLeftObj, ref_R, ref_G, ref_B);
            surfaceColorChanger.ChangeSurfaceTextureColor(RendererRightObj, ref_R, ref_G, ref_B);
        }
        else if (OddLocation == 2)
        {
            RGB_comp_corrected = surfaceColorChanger.ChangeSurfaceTextureColor(RendererLeftObj, comp_R, comp_G, comp_B);
            RGB_ref_corrected = surfaceColorChanger.ChangeSurfaceTextureColor(RendererTopObj, ref_R, ref_G, ref_B);
            surfaceColorChanger.ChangeSurfaceTextureColor(RendererRightObj, ref_R, ref_G, ref_B);
        }
        else if (OddLocation == 3)
        {
            RGB_comp_corrected = surfaceColorChanger.ChangeSurfaceTextureColor(RendererRightObj, comp_R, comp_G, comp_B);
            RGB_ref_corrected = surfaceColorChanger.ChangeSurfaceTextureColor(RendererTopObj, ref_R, ref_G, ref_B);
            surfaceColorChanger.ChangeSurfaceTextureColor(RendererLeftObj, ref_R, ref_G, ref_B);
        }

        // 📷 Take a screenshot after the first frame update (only once per iteration)
        if (takeScreenshotOfStimuli && !screenshotTaken)
        {
            yield return new WaitForEndOfFrame(); // Let one full frame pass with the stimuli visible
            string filename = Path.Combine(saveDirectory, $"Ref_R{ref_R}_G{ref_G}_B{ref_B}_Target_R{comp_R}_G{comp_G}_B{comp_B}.png");
            ScreenCapture.CaptureScreenshot(filename);
            screenshotTaken = true;  // Prevent multiple screenshots in the same iteration
            Debug.Log($"📷 Screenshot saved: {filename}");
            yield return new WaitForEndOfFrame();  // Ensure screenshot is fully processed
        }
        screenshotTaken = false;  // Reset flag for next iteration

        // 🍡 Keep the stimuli on screen for the duration of the stimulus presentation
        yield return new WaitForSeconds(durations["stimulus"]);

        // 🙈 Hide stimuli after presentation ends
        RendererTopObj.enabled   = false;
        RendererLeftObj.enabled  = false;
        RendererRightObj.enabled = false;

        // Convert RGB values to string format for logging and storage
        string RGB_ref_string = fileDataExtractor.ConvertRGBToString(ref_R, ref_G, ref_B);
        string RGB_comp_string = fileDataExtractor.ConvertRGBToString(comp_R, comp_G, comp_B);
        string RGB_ref_corrected_string = fileDataExtractor.ConvertRGBToString(RGB_ref_corrected.r, RGB_ref_corrected.g, RGB_ref_corrected.b);
        string RGB_comp_corrected_string = fileDataExtractor.ConvertRGBToString(RGB_comp_corrected.r, RGB_comp_corrected.g, RGB_comp_corrected.b);

        // 🎮 Capture the participant's response
        yield return StartCoroutine(GetResponse(OddLocation));

        // 💾 Log trial data, including stimulus parameters and participant response
        RegisterResponse(OddLocation, TrialInfo, RGB_ref_string, RGB_comp_string, 
                        RGB_ref_corrected_string, RGB_comp_corrected_string, 
                        codedAnswer, pressedKeysString, actualWait_preStim, responseTime);
    
        // ➕ Move to the next trial
        trialCounter++;
    }

    /// <summary>
    /// Waits for the participant's response and records the reaction time.
    /// Determines which key was pressed and encodes the response.
    /// OddLocation: The position of the odd (comparison) stimulus.
    public IEnumerator GetResponse(int OddLocation)
    {
        // Display response instruction (participant must press a key)
        displayText.text = "< ^ >";
        
        // 📷 Take a screenshot after the first frame update (only once per iteration)
        if (takeScreenshotOfStimuli && trialCounter == 1)
        {
            string filename = Path.Combine(saveDirectory, $"responseprobe.png");
            ScreenCapture.CaptureScreenshot(filename);
            yield return new WaitForEndOfFrame();  // Ensure screenshot is fully processed
        }

        // 🕒 Record the time when response waiting starts
        float responseStartTime = Time.time;

        // ⏳ Wait until one of the response keys is pressed, and store the key
        KeyCode pressedKey = KeyCode.None; // Variable to store the pressed key
        bool isGamepadPressed = false;

        yield return new WaitUntil(() =>
        {
            //check the keyboard
            if (Input.GetKeyDown(ResponseKeyLeft)) { pressedKey = ResponseKeyLeft; return true; }
            if (Input.GetKeyDown(ResponseKeyRight)) { pressedKey = ResponseKeyRight; return true; }
            if (Input.GetKeyDown(ResponseKeyTop)) { pressedKey = ResponseKeyTop; return true; }
            
            //check the gamepad
            if (gamepad != null)
            {
                if (gamepad.buttonWest.wasPressedThisFrame) { pressedKey = KeyCode.None; isGamepadPressed = true; return true; } //X button (left)
                if (gamepad.buttonNorth.wasPressedThisFrame) { pressedKey = KeyCode.None; isGamepadPressed = true; return true; } //Y button (top)
                if (gamepad.buttonEast.wasPressedThisFrame) { pressedKey = KeyCode.None; isGamepadPressed = true; return true; } //X button (right)
            }
            return false;
        });

        // ⏱️ Calculate the reaction time
        responseTime = Time.time - responseStartTime;

        // ⌨ Determine which key was pressed
        bool isKeyLeftPressed = pressedKey == ResponseKeyLeft;
        bool isKeyRightPressed = pressedKey == ResponseKeyRight;
        bool isKeyTopPressed = pressedKey == ResponseKeyTop;

        // 🎮 Determine gamepad response
        bool isGamepadXPressed = isGamepadPressed && gamepad.buttonWest.wasPressedThisFrame; // X → Left
        bool isGamepadYPressed = isGamepadPressed && gamepad.buttonNorth.wasPressedThisFrame; // Y → Top
        bool isGamepadBPressed = isGamepadPressed && gamepad.buttonEast.wasPressedThisFrame; // B → Right

        // 🧠 Compute the participant’s response based on the key press and odd stimulus location
        codedAnswer = trialHelper.CheckUserResponse(
            isKeyTopPressed || isGamepadYPressed,  // Y Button = Top Response
            isKeyLeftPressed || isGamepadXPressed, // X Button = Left Response
            isKeyRightPressed || isGamepadBPressed, // B Button = Right Response
            OddLocation
        );

        // Convert the pressed key data into a formatted string for logging
        List<bool> pressedKeys = new List<bool>
        {
            isKeyTopPressed || isGamepadYPressed,
            isKeyLeftPressed || isGamepadXPressed,
            isKeyRightPressed || isGamepadBPressed
        };
        pressedKeysString = string.Join("_", pressedKeys.Select(b => b ? "1" : "0"));
    }

    /// <summary>
    /// Logs the participant's response and stores trial-related data.
    /// This function ensures that AEPsych receives the response immediately 
    /// to compute the next trial efficiently while also recording the data for analysis.
    /// trialCounter       : Current trial number (starting from 0)
    /// trialInfo          : Information about the trial (e.g., Trial_10_MOCS_5)
    /// oddLocation        : Position of the odd stimulus (1 = Top, 2 = Left, 3 = Right)
    /// pressedKeys        : String representation of the response keys pressed (e.g., "1_0_0", "0_1_0", "0_0_1")
    /// RGB_ref            : RGB values of the reference stimulus before correction
    /// RGB_comp           : RGB values of the comparison stimulus before correction
    /// RGB_ref_corrected  : Corrected RGB values of the reference stimulus
    /// RGB_comp_corrected : Corrected RGB values of the comparison stimulus
    /// codedResp          : Binary-coded response (1 = Correct, 0 = Incorrect)
    /// RT                 : Reaction time (in seconds)
    /// actual_wait_preStim: The time between the disappearance of '+' and stimulus presentation
    /// LUT                : look-up table for gamma correction
    private void RegisterResponse(int OddLocation, string TrialInfo, string ref_RGB, string comp_RGB, 
        string ref_RGB_corrected, string comp_RGB_corrected, float binaryResp, string pressedKeys, 
        float actual_wait_preStim, float respTime)
    {
        // 💾 Log response immediately for real-time trial adaptation
        // This allows AEPsych to compute the next trial placement as soon as possible.
        fileDataExtractor.WriteResponseToFile(initFilePath, TrialInfo, ref_RGB, comp_RGB, codedAnswer);

        // 💾 Append trial data for long-term storage & analysis
        // This ensures the collected responses are saved in a structured CSV file.
        AppendTrialData(
            trialCounter.ToString(),  
            TrialInfo,                
            OddLocation.ToString(),   
            pressedKeys,              
            ref_RGB,                  
            comp_RGB,                 
            ref_RGB_corrected,        
            comp_RGB_corrected,       
            binaryResp.ToString(),   
            actual_wait_preStim.ToString(),
            respTime.ToString()   
        );
    }

    private void createTrialData()
    {
        // Generate timestamp in the format YYYY-MM-DD_HH-MM-SS
        string timeStamp = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");

        // Construct the file path with timestamp
        csvFilePath = Path.Combine(saveDirectory, $"Unity_trial_data_{filePrefix}_{timeStamp}.csv");

        // If the file does not exist, write the header
        if (!File.Exists(csvFilePath))
        {
            using (StreamWriter sw = new StreamWriter(csvFilePath, true))
            {
                sw.WriteLine("TrialCounter,TrialInfo, OddLocation, PressedKey(Top-Left-Right), Ref, Comp, "+
                 "Ref_Corrected,Comp_Corrected, binaryResp, actual_wait_preStim, RT, " +
                  $"subID_{subject_id.ToString()}, subInit_{subject_init}, sessNum_{session_today.ToString()}, "+
                  $"seed_{subject_seed.ToString()}, LUT_{gammaCorrectionFile}");
            }
        }
    }

    private void AppendTrialData(
        string trialCounter, string trialInfo, string oddLocation, string pressedKeys, 
        string RGB_ref, string RGB_comp, string RGB_ref_corrected, string RGB_comp_corrected, 
        string codedResp, string actual_wait_preStim, string RT)
    {
        using (StreamWriter sw = new StreamWriter(csvFilePath, true))
        {
            // Write to CSV file with better formatting (split into two lines for readability)
            sw.WriteLine(
                $"{trialCounter}, {trialInfo}, {oddLocation}, {pressedKeys}, {RGB_ref}, {RGB_comp}, " +
                $"{RGB_ref_corrected}, {RGB_comp_corrected}, {codedResp}, {actual_wait_preStim}, {RT}"
            );
        }
    }

    public void ConnectGamepads()
    {
        var gamepads = Gamepad.all;

        if (gamepads.Count > 0)
        {
            // Track only the first connected gamepad (assuming max 1 gamepad)
            gamepad = gamepads[0];
            Debug.Log($"🎮 Gamepad Connected: {gamepad.displayName}");
        }
        else
        {
            // No gamepad found, update flag and set gamepad to null
            gamepad = null;
            Debug.Log($"No gamepad found. We will use a keyboard for registering responses.");
        }
    }
}



