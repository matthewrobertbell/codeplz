use rayon::prelude::*;
use std::cmp::min;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{anyhow, bail, Context};
use askama::Template;
use aws_sdk_bedrockruntime::{
    operation::converse::ConverseOutput,
    types::{ContentBlock, ConversationRole, Message},
    Client as BedrockClient,
};
use aws_sdk_sts::Client as StsClient;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::{
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use globset::{Glob, GlobSetBuilder};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use similar::{ChangeTag, TextDiff};
use tiktoken_rs::cl100k_base;
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Port number to run the server on
    #[arg(short, long, default_value_t = 3000)]
    port: u16,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
enum LLMModel {
    #[default]
    OpenAIGPT4o,
    Claude35SonnetBedrock,
}

impl LLMModel {
    fn max_tokens(&self) -> usize {
        match self {
            LLMModel::OpenAIGPT4o => 100_000,
            LLMModel::Claude35SonnetBedrock => 180_000,
        }
    }
}
fn default_true() -> bool {
    true
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
struct Config {
    include_paths: Vec<String>,
    exclude_paths: Vec<String>,
    commands: Vec<String>,
    #[serde(default = "default_true")]
    use_gitignore: bool,
    system_prompt: Option<String>,
    model: LLMModel,
    maximum_context_tokens: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
}

#[derive(Serialize, Deserialize, Debug)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Deserialize, Debug)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Deserialize, Debug)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

#[derive(Debug, Deserialize)]
struct LLMResponse {
    explanation: String,
    changes: Vec<Change>,
    conclusion: String,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
enum LineOrLines {
    Line(String),
    Lines(Vec<String>),
}

impl LineOrLines {
    fn lines(&self) -> Vec<String> {
        match self {
            LineOrLines::Line(line) => vec![line.clone()],
            LineOrLines::Lines(lines) => lines.clone(),
        }
    }

    fn len(&self) -> usize {
        match self {
            LineOrLines::Line(_) => 1,
            LineOrLines::Lines(lines) => lines.len(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "command", rename_all = "SCREAMING_SNAKE_CASE")]
enum LLMCommand {
    InsertAfter {
        insert_lines: LineOrLines,
        marker_lines: LineOrLines,
    },
    InsertBefore {
        insert_lines: LineOrLines,
        marker_lines: LineOrLines,
    },
    Delete {
        delete_lines: LineOrLines,
    },
    CreateFile {
        new_lines: LineOrLines,
    },
    RenameFile {
        new_filename: PathBuf,
    },
    DeleteFile,
}

#[derive(Debug, Deserialize, Serialize)]
struct Change {
    filename: PathBuf,
    #[serde(flatten)]
    command: LLMCommand,
    reason: String,
}

#[derive(Debug, Serialize)]
struct ChangeWithDiff {
    #[serde(flatten)]
    change: Change,
    diff: String,
}

#[derive(Clone)]
struct AppState {
    config: Config,
}

impl AppState {
    fn new(config: Config) -> Self {
        AppState { config }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let config_path = "codeplz.json";
    let config = match std::fs::read_to_string(config_path) {
        Ok(content) => serde_json::from_str::<Config>(&content)?,
        Err(_) => {
            let default_config = Config::default();
            let config_json = serde_json::to_string_pretty(&default_config).unwrap();
            std::fs::write(config_path, config_json).context("Failed to create config file")?;
            default_config
        }
    };
    println!("Loaded config: {:?}", config);

    // Validate credentials based on the selected model
    match config.model {
        LLMModel::Claude35SonnetBedrock => validate_aws_credentials().await?,
        LLMModel::OpenAIGPT4o => validate_openai_api_key()?,
    }

    // Create AppState with the updated config
    let state = AppState::new(config);

    // Update the main function to include the new route
    let app = Router::new()
        .route("/", get(index))
        .route("/prompt", post(prompt))
        .route("/apply_changes", post(apply_changes))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", args.port)).await?;
    println!("Server running on http://127.0.0.1:{}", args.port);
    axum::serve(listener, app).await?;

    Ok(())
}

async fn validate_aws_credentials() -> anyhow::Result<()> {
    println!("Validating AWS credentials...");
    let aws_config = aws_config::load_from_env().await;
    let sts_client = StsClient::new(&aws_config);

    // Attempt to get the caller identity to validate credentials
    sts_client
        .get_caller_identity()
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to validate AWS credentials: {}", e))?;

    println!("AWS credentials validated successfully.");
    Ok(())
}

// Add this new function
fn validate_openai_api_key() -> anyhow::Result<()> {
    match std::env::var("OPENAI_API_KEY") {
        Ok(_) => {
            println!("OpenAI API key found in environment variables.");
            Ok(())
        }
        Err(_) => {
            anyhow::bail!("OPENAI_API_KEY environment variable not set. Please set it before running the application with the OpenAI GPT-4 model.")
        }
    }
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {
    code_path: String,
    model_name: String,
    project_name_last: String,
}

async fn index(State(state): State<AppState>) -> IndexTemplate {
    let code_path = std::env::current_dir()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|_| String::from("Unknown"));
    let project_name_last = std::env::current_dir()
        .map(|path| {
            path.file_name()
                .and_then(|name| name.to_str().map(|s| s.to_string()))
        })
        .unwrap_or_else(|_| Some(String::from("Unknown")))
        .unwrap_or_else(|| String::from("Unknown"))
        .split('/')
        .last()
        .unwrap_or("Unknown")
        .to_string();
    let model_name = match state.config.model {
        LLMModel::OpenAIGPT4o => "OpenAI GPT-4",
        LLMModel::Claude35SonnetBedrock => "Claude 3.5 Sonnet (Bedrock)",
    }
    .to_string();
    IndexTemplate {
        code_path,
        model_name,
        project_name_last,
    }
}

// Add this new struct
#[derive(Deserialize)]
struct PromptRequest {
    prompt: String,
}

#[derive(Serialize)]
struct ProcessedFile {
    name: String,
    token_count: usize,
}

#[derive(Serialize)]
struct PromptResponse {
    explanation: String,
    changes: Vec<ChangeWithDiff>,
    conclusion: String,
    input_token_count: usize,
    output_token_count: usize,
    processed_files: Vec<ProcessedFile>,
}

async fn select_relevant_files(
    files_and_tokens: &[(PathBuf, usize)],
    user_prompt: &str,
    config: &Config,
) -> Result<Vec<PathBuf>, (StatusCode, String)> {
    let file_list = files_and_tokens
        .iter()
        .map(|(path, tokens)| format!("{}: {} tokens", path.display(), tokens))
        .collect::<Vec<_>>()
        .join("\n");

    if file_list.is_empty() {
        return Ok(Vec::new());
    }

    let pre_prompt = format!(
        "Given the following list of files and their token counts, and the user's request, \
        select the most relevant files for completing the task. Return only the file paths, \
        one per line, without any additional text or explanation.\n\n\
        If there are a lot less than 100,000 tokens in the list, return files that have a chance of being relevant.\
        If there are a lot more than 100,000 tokens in the list, return only the most relevant files.\
        \nFiles:\n{}\n\nUser request: {}\n\nRelevant files:",
        file_list, user_prompt
    );

    let response = match config.model {
        LLMModel::OpenAIGPT4o => {
            call_openai_gpt4("You are a helpful assistant.", &pre_prompt).await
        }
        LLMModel::Claude35SonnetBedrock => {
            call_claude_bedrock("You are a helpful assistant.", &pre_prompt).await
        }
    }?;

    dbg!(&response);

    let relevant_files = response
        .lines()
        .filter_map(|line| {
            let path = PathBuf::from(line.trim());
            files_and_tokens
                .iter()
                .find(|(p, _)| p == &path)
                .map(|(_, _)| path)
        })
        .collect();

    dbg!(&relevant_files);

    Ok(relevant_files)
}

// Modify the prompt function
async fn prompt(
    State(state): State<AppState>,
    Json(request): Json<PromptRequest>,
) -> Result<Json<PromptResponse>, (StatusCode, String)> {
    let start_time = Instant::now();

    let config = &state.config;
    let maximum_context_tokens = config
        .maximum_context_tokens
        .unwrap_or_else(|| config.model.max_tokens());

    // Load all files
    let load_files_start = Instant::now();
    let (files_and_tokens, _) = load_files(config, usize::MAX).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to load files: {}", e),
        )
    })?;
    let load_files_duration = load_files_start.elapsed();
    println!(
        "Load files duration: {} ms",
        load_files_duration.as_millis()
    );

    // Select relevant files
    let select_files_start = Instant::now();
    let relevant_files = select_relevant_files(&files_and_tokens, &request.prompt, config).await?;
    let select_files_duration = select_files_start.elapsed();
    println!(
        "Select files duration: {} ms",
        select_files_duration.as_millis()
    );

    // Filter files_and_tokens to include only relevant files
    let relevant_files_and_tokens: Vec<(PathBuf, usize)> = files_and_tokens
        .into_iter()
        .filter(|(path, _)| relevant_files.contains(path))
        .collect();

    // Generate context using only relevant files
    let make_context_start = Instant::now();
    let (context, processed_files) =
        make_context(maximum_context_tokens, config, &relevant_files_and_tokens).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to make context: {}", e),
            )
        })?;
    let make_context_duration = make_context_start.elapsed();
    println!(
        "Make context duration: {} ms",
        make_context_duration.as_millis()
    );

    // Get the system prompt
    let system_prompt = config
        .system_prompt
        .clone()
        .unwrap_or_else(|| include_str!("default_system_prompt.txt").to_string());

    // Combine system prompt, context, and user prompt
    let full_prompt = format!("{}\n\nUser request: {}", context, request.prompt);

    // Calculate input token count
    let bpe = cl100k_base().unwrap();
    let input_token_count = bpe.encode_with_special_tokens(&full_prompt).len();

    // Call LLM based on the configured model
    let llm_call_start = Instant::now();
    let llm_response = match config.model {
        LLMModel::OpenAIGPT4o => call_openai_gpt4(&system_prompt, &full_prompt).await?,
        LLMModel::Claude35SonnetBedrock => {
            call_claude_bedrock(&system_prompt, &full_prompt).await?
        }
    }
    .replace("```json", "")
    .replace("```", "");
    let llm_call_duration = llm_call_start.elapsed();
    println!("LLM call duration: {} ms", llm_call_duration.as_millis());

    // Check if the response starts with '{'
    let llm_response = if !llm_response.trim_start().starts_with('{') {
        llm_response
            .split_once('{')
            .map(|(_, json_part)| format!("{{{}", json_part))
            .unwrap_or(llm_response)
    } else {
        llm_response
    };

    dbg!(&llm_response);

    // Calculate output token count
    let output_token_count = bpe.encode_with_special_tokens(&llm_response).len();

    // Parse LLM response
    let parse_response_start = Instant::now();
    let llm_data: LLMResponse = serde_json::from_str(&llm_response).map_err(|e| {
        eprintln!("Error parsing LLM response: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Error processing request".to_string(),
        )
    })?;
    let parse_response_duration = parse_response_start.elapsed();
    println!(
        "Parse response duration: {} ms",
        parse_response_duration.as_millis()
    );

    // Generate diffs for each validated change
    let generate_diffs_start = Instant::now();
    let changes_with_diff = validate_changes(llm_data.changes)
        .into_iter()
        .map(|change| {
            let diff =
                generate_diff(&change).unwrap_or_else(|_| String::from("Unable to generate diff"));
            ChangeWithDiff { change, diff }
        })
        .collect::<Vec<ChangeWithDiff>>();
    let generate_diffs_duration = generate_diffs_start.elapsed();
    println!(
        "Generate diffs duration: {} ms",
        generate_diffs_duration.as_millis()
    );

    let total_duration = start_time.elapsed();
    println!("Total duration: {} ms", total_duration.as_millis());

    let response = PromptResponse {
        explanation: llm_data.explanation,
        changes: changes_with_diff,
        conclusion: llm_data.conclusion,
        input_token_count,
        output_token_count,
        processed_files,
    };

    Ok(Json(response))
}

async fn call_openai_gpt4(
    system_prompt: &str,
    full_prompt: &str,
) -> Result<String, (StatusCode, String)> {
    let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "OPENAI_API_KEY environment variable not set".to_string(),
        )
    })?;

    let client = Client::new();
    let request = OpenAIRequest {
        model: "gpt-4o-2024-08-06".to_string(),
        messages: vec![
            OpenAIMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            OpenAIMessage {
                role: "user".to_string(),
                content: full_prompt.to_string(),
            },
        ],
    };

    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request)
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to send request to OpenAI API: {}", e),
            )
        })?;

    let response_text = response.text().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to read response from OpenAI API: {}", e),
        )
    })?;

    dbg!(&response_text);

    let openai_response: OpenAIResponse = serde_json::from_str(&response_text).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to parse OpenAI API response: {}", e),
        )
    })?;

    openai_response
        .choices
        .first()
        .map(|choice| choice.message.content.clone())
        .ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "No response from OpenAI API".to_string(),
        ))
}

fn validate_changes(changes: Vec<Change>) -> Vec<Change> {
    changes
        .into_iter()
        .filter_map(|change| match validate_change(&change) {
            Ok(_) => Some(change),
            Err(e) => {
                eprintln!("Failed to validate change: {:?}", e);
                None
            }
        })
        .collect()
}

fn validate_change(change: &Change) -> anyhow::Result<()> {
    // First, check if the file is in the current directory
    if !is_file_in_current_directory(&change.filename) {
        bail!(
            "File is outside the current directory: {:?}",
            change.filename
        );
    }

    match &change.command {
        LLMCommand::CreateFile { .. } => {
            // Remove the check for existing files
        }
        LLMCommand::RenameFile { new_filename } => {
            // Check if the source file exists
            if !change.filename.exists() {
                bail!("Source file does not exist: {:?}", change.filename);
            }
            // Also check if the new filename is in the current directory
            if !is_file_in_current_directory(new_filename) {
                bail!(
                    "New filename is outside the current directory: {:?}",
                    new_filename
                );
            }
        }
        LLMCommand::DeleteFile => {
            // Check if the file exists
            if !change.filename.exists() {
                bail!("File does not exist: {:?}", change.filename);
            }
        }
        LLMCommand::InsertAfter { marker_lines, .. }
        | LLMCommand::InsertBefore { marker_lines, .. }
        | LLMCommand::Delete {
            delete_lines: marker_lines,
        } => {
            // Check if the file exists and the marker lines can be found
            let file_content = std::fs::read_to_string(&change.filename)?;
            let file_lines: Vec<String> = file_content.lines().map(String::from).collect();
            if find_in_file_lines(&file_lines, &marker_lines.lines()).is_none() {
                bail!("Failed to find marker lines in file: {:?}", change.filename);
            }
        }
    }
    Ok(())
}

fn generate_diff(change: &Change) -> anyhow::Result<String> {
    match &change.command {
        LLMCommand::CreateFile { new_lines } => {
            // For new files, show all lines as added
            Ok(new_lines
                .lines()
                .into_iter()
                .map(|line| format!("+{}\n", line))
                .fold(String::new(), |acc, line| acc + &line))
        }
        _ => {
            let file_path = &change.filename;
            let old_content = std::fs::read_to_string(file_path)?;
            let new_content = apply_change_to_content(&old_content, change)?;

            // Trim the last line of both old and new content
            let old_content_trimmed = old_content.trim_end();
            let new_content_trimmed = new_content.trim_end();

            let diff = TextDiff::from_lines(old_content_trimmed, new_content_trimmed);
            let mut diff_output = String::new();
            let mut unchanged_lines = Vec::new();

            for change in diff.iter_all_changes() {
                match change.tag() {
                    ChangeTag::Equal => {
                        unchanged_lines.push(change.to_string());
                        if unchanged_lines.len() > 10 {
                            unchanged_lines.remove(0);
                        }
                    }
                    ChangeTag::Delete | ChangeTag::Insert => {
                        // Output up to 5 unchanged lines before the change
                        let start = unchanged_lines.len().saturating_sub(5);
                        for line in &unchanged_lines[start..] {
                            diff_output.push_str(&format!(" {}", line));
                        }
                        unchanged_lines.clear();

                        // Output the changed line
                        let sign = if change.tag() == ChangeTag::Delete {
                            "-"
                        } else {
                            "+"
                        };
                        diff_output.push_str(&format!("{}{}", sign, change));
                    }
                }
            }

            // Output up to 5 unchanged lines after the last change
            let end = unchanged_lines.len().min(5);
            for line in &unchanged_lines[..end] {
                diff_output.push_str(&format!(" {}", line));
            }

            Ok(diff_output)
        }
    }
}

#[derive(Deserialize)]
struct ApplyChangesRequest {
    changes: Vec<Change>,
}

#[derive(Serialize)]
struct ApplyChangesResponse {
    results: Vec<ChangeResult>,
}

#[derive(Serialize)]
struct ChangeResult {
    filename: String,
    success: bool,
    message: String,
}

async fn apply_changes(Json(request): Json<ApplyChangesRequest>) -> impl IntoResponse {
    let mut results = Vec::new();

    for change in request.changes {
        let result = apply_change(&change);
        results.push(result);
    }

    Json(ApplyChangesResponse { results })
}

fn apply_change(change: &Change) -> ChangeResult {
    match &change.command {
        LLMCommand::CreateFile { new_lines } => {
            let file_path = Path::new(&change.filename);
            if let Some(parent) = file_path.parent() {
                if let Err(e) = fs::create_dir_all(parent) {
                    return ChangeResult {
                        filename: change.filename.to_string_lossy().into_owned(),
                        success: false,
                        message: format!("Failed to create directory: {}", e),
                    };
                }
            }
            match fs::write(file_path, new_lines.lines().join("\n")) {
                Ok(_) => ChangeResult {
                    filename: change.filename.to_string_lossy().into_owned(),
                    success: true,
                    message: format!("Created file and inserted {} lines", new_lines.len()),
                },
                Err(e) => ChangeResult {
                    filename: change.filename.to_string_lossy().into_owned(),
                    success: false,
                    message: format!("Failed to create file: {}", e),
                },
            }
        }
        LLMCommand::RenameFile { new_filename } => {
            match fs::rename(&change.filename, new_filename) {
                Ok(_) => ChangeResult {
                    filename: change.filename.to_string_lossy().into_owned(),
                    success: true,
                    message: format!("Renamed file to {}", new_filename.to_string_lossy()),
                },
                Err(e) => ChangeResult {
                    filename: change.filename.to_string_lossy().into_owned(),
                    success: false,
                    message: format!("Failed to rename file: {}", e),
                },
            }
        }
        LLMCommand::DeleteFile => match fs::remove_file(&change.filename) {
            Ok(_) => ChangeResult {
                filename: change.filename.to_string_lossy().into_owned(),
                success: true,
                message: "Deleted file".to_string(),
            },
            Err(e) => ChangeResult {
                filename: change.filename.to_string_lossy().into_owned(),
                success: false,
                message: format!("Failed to delete file: {}", e),
            },
        },
        _ => {
            let file_path = Path::new(&change.filename);
            match fs::read_to_string(file_path) {
                Ok(content) => match apply_change_to_content(&content, change) {
                    Ok(new_content) => match fs::write(file_path, new_content) {
                        Ok(_) => ChangeResult {
                            filename: change.filename.to_string_lossy().into_owned(),
                            success: true,
                            message: "Applied changes successfully".to_string(),
                        },
                        Err(e) => ChangeResult {
                            filename: change.filename.to_string_lossy().into_owned(),
                            success: false,
                            message: format!("Failed to write changes: {}", e),
                        },
                    },
                    Err(e) => ChangeResult {
                        filename: change.filename.to_string_lossy().into_owned(),
                        success: false,
                        message: format!("Failed to apply changes: {}", e),
                    },
                },
                Err(e) => ChangeResult {
                    filename: change.filename.to_string_lossy().into_owned(),
                    success: false,
                    message: format!("Failed to read file: {}", e),
                },
            }
        }
    }
}

fn apply_change_to_content(content: &str, change: &Change) -> anyhow::Result<String> {
    let lines: Vec<String> = content.lines().map(String::from).collect();
    let mut new_lines = lines.clone();

    match &change.command {
        LLMCommand::CreateFile {
            new_lines: create_lines,
        } => {
            // For new files, return the new content directly
            return Ok(create_lines.lines().join("\n"));
        }
        LLMCommand::InsertAfter {
            marker_lines,
            insert_lines,
        } => {
            let marker_lines: Vec<String> =
                marker_lines.lines().into_iter().map(String::from).collect();
            if let Some(index) = find_in_file_lines(&lines, &marker_lines) {
                let mut insert = insert_lines
                    .lines()
                    .into_iter()
                    .map(String::from)
                    .collect::<Vec<String>>();

                // Remove any lines from insert that match the end of marker_lines
                let overlap = marker_lines
                    .iter()
                    .rev()
                    .zip(insert.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                insert.drain(0..overlap);

                new_lines.splice(
                    index + marker_lines.len()..index + marker_lines.len(),
                    insert,
                );
            }
        }
        LLMCommand::InsertBefore {
            marker_lines,
            insert_lines,
        } => {
            let marker_lines: Vec<String> =
                marker_lines.lines().into_iter().map(String::from).collect();
            if let Some(index) = find_in_file_lines(&lines, &marker_lines) {
                let mut insert = insert_lines
                    .lines()
                    .into_iter()
                    .map(String::from)
                    .collect::<Vec<String>>();

                // Remove any lines from insert that match the start of marker_lines
                let overlap = marker_lines
                    .iter()
                    .zip(insert.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                insert.drain(0..overlap);

                new_lines.splice(index..index, insert);
            }
        }
        LLMCommand::Delete { delete_lines } => {
            if let Some(index) = find_in_file_lines(&lines, &delete_lines.lines()) {
                new_lines.drain(index..index + delete_lines.lines().len());
            }
        }
        LLMCommand::RenameFile { .. } | LLMCommand::DeleteFile => {}
    }

    Ok(new_lines.join("\n"))
}

async fn call_claude_bedrock(
    system_prompt: &str,
    prompt: &str,
) -> Result<String, (StatusCode, String)> {
    let aws_config = aws_config::load_from_env().await;
    let bedrock_client = BedrockClient::new(&aws_config);

    let full_prompt = format!("{}\n\n{}", system_prompt, prompt);

    let response = bedrock_client
        .converse()
        .model_id("anthropic.claude-3-5-sonnet-20240620-v1:0")
        .messages(
            Message::builder()
                .role(ConversationRole::User)
                .content(ContentBlock::Text(full_prompt))
                .build()
                .map_err(|_| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "failed to build message".to_string(),
                    )
                })?,
        )
        .send()
        .await
        .inspect_err(|e| {
            eprintln!("Error: {}", e);
        })
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    dbg!(&response);

    get_bedrock_converse_output_text(response).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))
}

fn get_bedrock_converse_output_text(output: ConverseOutput) -> Result<String, String> {
    let text = output
        .output()
        .ok_or("no output".to_string())?
        .as_message()
        .map_err(|_| "output not a message".to_string())?
        .content()
        .first()
        .ok_or("no content in message".to_string())?
        .as_text()
        .map_err(|_| "content is not text".to_string())?
        .to_string();
    Ok(text)
}

fn is_file_in_current_directory(path: &Path) -> bool {
    path.is_relative() && !path.starts_with("..")
}

fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for (i, row) in matrix.iter_mut().enumerate() {
        row[0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    for (i, c1) in s1.chars().enumerate() {
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            matrix[i + 1][j + 1] = min(
                min(matrix[i][j + 1] + 1, matrix[i + 1][j] + 1),
                matrix[i][j] + cost,
            );
        }
    }

    matrix[len1][len2]
}

fn find_in_file_lines(file_lines: &[String], needle: &[String]) -> Option<usize> {
    let non_empty_needle: Vec<_> = needle
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if non_empty_needle.is_empty() {
        return Some(0);
    }

    let needle_joined = non_empty_needle.join("\n");
    let needle_len = needle_joined.chars().count();
    let mut best_match = None;
    let mut min_distance = usize::MAX;

    for (i, window) in file_lines.windows(needle.len()).enumerate() {
        let window_joined = window
            .iter()
            .map(|s| s.trim())
            .collect::<Vec<_>>()
            .join("\n");
        let distance = levenshtein_distance(&needle_joined, &window_joined);

        if distance < min_distance {
            min_distance = distance;
            best_match = Some(i);
        }

        if distance == 0 {
            break; // Exact match found
        }
    }

    // Check if the best match meets the 95% similarity threshold
    if let Some(i) = best_match {
        let similarity = 1.0 - (min_distance as f64 / needle_len as f64);

        if similarity >= 0.95 {
            return Some(i);
        } else {
            println!("Best match similarity: {}", similarity);
            println!(
                "Best match: {:?}",
                &file_lines[i..min(i + needle.len(), file_lines.len())]
            );
            println!("Marker lines: {:?}", needle);
        }
    }

    None
}

fn make_context(
    maximum_context_tokens: usize,
    config: &Config,
    files_and_tokens: &[(PathBuf, usize)],
) -> anyhow::Result<(String, Vec<ProcessedFile>)> {
    let mut content = String::new();
    let mut processed_files = Vec::new();
    let mut total_tokens = 0;
    let bpe = cl100k_base().unwrap();

    for (path, token_count) in files_and_tokens {
        let file_content = fs::read_to_string(path)?;
        let path_str = path.to_str().ok_or_else(|| anyhow!("Invalid path"))?;
        let path_str = path_str.strip_prefix("./").unwrap_or(path_str);

        let file_context = format!(
            r#"File name: "{}"

File contents: """
{}"""
----------

"#,
            path_str, file_content
        );
        content.push_str(&file_context);
        total_tokens += token_count;
        processed_files.push(ProcessedFile {
            name: path_str.to_string(),
            token_count: *token_count,
        });
    }

    // Process commands
    for cmd in &config.commands {
        let output = execute_command(cmd)?;
        let command_context = format!(
            r#"Command: "{}"

Command output: """
{}"""
----------

"#,
            cmd, output
        );
        let command_token_count = bpe.encode_with_special_tokens(&command_context).len();

        if total_tokens + command_token_count <= maximum_context_tokens {
            content.push_str(&command_context);
            total_tokens += command_token_count;
            processed_files.push(ProcessedFile {
                name: format!("Command: {}", cmd),
                token_count: command_token_count,
            });
        } else {
            break;
        }
    }

    let content = content.replace('\n', "\r\n");

    println!("Success loading context! Token count: {}", total_tokens);
    Ok((content, processed_files))
}

fn execute_command(cmd: &str) -> anyhow::Result<String> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .output()
        .with_context(|| format!("Failed to execute command: {}", cmd))?;

    let exit_status = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    Ok(format!(
        "Exit Status: {}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}",
        exit_status, stdout, stderr
    ))
}

fn load_files(
    config: &Config,
    maximum_context_tokens: usize,
) -> anyhow::Result<(Vec<(PathBuf, usize)>, usize)> {
    let bpe = cl100k_base().unwrap();

    let mut include_builder = GlobSetBuilder::new();
    let mut exclude_builder = GlobSetBuilder::new();

    for glob in &config.include_paths {
        include_builder
            .add(Glob::new(glob).map_err(|e| anyhow!("Invalid include glob pattern: {}", e))?);
    }

    for glob in &config.exclude_paths {
        exclude_builder
            .add(Glob::new(glob).map_err(|e| anyhow!("Invalid exclude glob pattern: {}", e))?);
    }

    if config.include_paths.is_empty() {
        include_builder.add(Glob::new("**/*").unwrap());
    }

    if config.exclude_paths.is_empty() {
        exclude_builder.add(Glob::new(".git/**/*").unwrap());
        exclude_builder.add(Glob::new("codeplz.json").unwrap());
    }

    let include_set = include_builder
        .build()
        .map_err(|e| anyhow!("Failed to build include globset: {}", e))?;
    let exclude_set = exclude_builder
        .build()
        .map_err(|e| anyhow!("Failed to build exclude globset: {}", e))?;
    let entries: Vec<_> = WalkDir::new(".")
        .into_iter()
        .filter_map(|e| e.ok())
        .collect();

    let mut files_and_tokens: Vec<_> = entries
        .par_iter()
        .filter_map(|entry| {
            let path = entry.path().to_path_buf();
            if path.is_dir() {
                return None;
            }

            let path_str = path.to_str()?;
            let path_str = path_str.strip_prefix("./").unwrap_or(path_str);

            if include_set.is_match(path_str) && !exclude_set.is_match(path_str) {
                match fs::read_to_string(&path) {
                    Ok(content) => {
                        let file_context = format!(
                            r#"File name: "{}"

File contents: """
{}
"""#,
                            path_str,
                            content.replace("\n", "\r\n")
                        );
                        let file_token_count = bpe.encode_with_special_tokens(&file_context).len();

                        Some((path, file_token_count))
                    }
                    Err(e) => {
                        if e.kind() != std::io::ErrorKind::InvalidData {
                            eprintln!("Failed to read file: {}", path_str);
                        }
                        None
                    }
                }
            } else {
                None
            }
        })
        .collect();

    files_and_tokens.par_sort_by(|a, b| a.0.cmp(&b.0));

    // Check for language-specific files and add exclude patterns
    let mut has_rust = false;
    let mut has_python = false;
    let mut has_js_ts = false;

    for (path, _) in &files_and_tokens {
        let path_str = path.to_str().unwrap_or("");
        if path_str.ends_with(".rs") {
            has_rust = true;
        } else if path_str.ends_with(".py") {
            has_python = true;
        } else if path_str.ends_with(".js") || path_str.ends_with(".ts") {
            has_js_ts = true;
        }
    }

    if has_rust {
        exclude_builder.add(Glob::new("**/target/**").unwrap());
        exclude_builder.add(Glob::new("**/Cargo.lock").unwrap());
    }
    if has_python {
        exclude_builder.add(Glob::new("**/__pycache__/**").unwrap());
        exclude_builder.add(Glob::new("**/*.pyc").unwrap());
    }
    if has_js_ts {
        exclude_builder.add(Glob::new("**/node_modules/**").unwrap());
        exclude_builder.add(Glob::new("**/package-lock.json").unwrap());
    }

    // Rebuild the exclude set with the new patterns
    let exclude_set = exclude_builder
        .build()
        .map_err(|e| anyhow!("Failed to build exclude globset: {}", e))?;

    // Filter files again with the updated exclude set
    files_and_tokens.retain(|(path, _)| {
        let path_str = path.to_str().unwrap_or("");
        let path_str = path_str.strip_prefix("./").unwrap_or(path_str);
        !exclude_set.is_match(path_str)
    });

    let mut cumulative_tokens = 0;
    let files_and_tokens: Vec<_> = files_and_tokens
        .into_iter()
        .take_while(|(_, token_count)| {
            cumulative_tokens += token_count;
            cumulative_tokens <= maximum_context_tokens
        })
        .collect();

    let total_tokens = files_and_tokens.iter().map(|(_, count)| *count).sum();

    Ok((files_and_tokens, total_tokens))
}
