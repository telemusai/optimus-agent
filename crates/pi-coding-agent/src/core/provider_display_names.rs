//! Port of packages/coding-agent/src/core/provider-display-names.ts

use indexmap::IndexMap;
use std::sync::OnceLock;

/// `BUILT_IN_PROVIDER_DISPLAY_NAMES` - declaration order is preserved.
pub fn built_in_provider_display_names() -> &'static IndexMap<String, String> {
    static NAMES: OnceLock<IndexMap<String, String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let pairs: [(&str, &str); 35] = [
            ("anthropic", "Anthropic"),
            ("amazon-bedrock", "Amazon Bedrock"),
            ("azure-openai-responses", "Azure OpenAI Responses"),
            ("cerebras", "Cerebras"),
            ("cloudflare-ai-gateway", "Cloudflare AI Gateway"),
            ("cloudflare-workers-ai", "Cloudflare Workers AI"),
            ("deepseek", "DeepSeek"),
            ("dgx-deepseek41", "DGX DeepSeek 4.1"),
            ("dgx-glm53", "DGX GLM 5.3"),
            ("dgx-k3", "DGX K3"),
            ("fireworks", "Fireworks"),
            ("google", "Google Gemini"),
            ("google-vertex", "Google Vertex AI"),
            ("groq", "Groq"),
            ("huggingface", "Hugging Face"),
            ("kimi-coding", "Kimi For Coding"),
            ("kiro", "Kiro"),
            ("mistral", "Mistral"),
            ("minimax", "MiniMax"),
            ("minimax-cn", "MiniMax (China)"),
            ("moonshotai", "Moonshot AI"),
            ("moonshotai-cn", "Moonshot AI (China)"),
            ("opencode", "OpenCode Zen"),
            ("opencode-go", "OpenCode Go"),
            ("openai", "OpenAI"),
            ("openrouter", "OpenRouter"),
            ("prime-agent-traces", "Prime Agent Traces"),
            ("prime-inference", "Prime Inference"),
            ("vercel-ai-gateway", "Vercel AI Gateway"),
            ("xai", "xAI"),
            ("zai", "ZAI"),
            ("xiaomi", "Xiaomi MiMo"),
            ("xiaomi-token-plan-cn", "Xiaomi MiMo Token Plan (China)"),
            ("xiaomi-token-plan-ams", "Xiaomi MiMo Token Plan (Amsterdam)"),
            ("xiaomi-token-plan-sgp", "Xiaomi MiMo Token Plan (Singapore)"),
        ];
        let mut map = IndexMap::new();
        for (key, value) in pairs {
            map.insert(key.to_string(), value.to_string());
        }
        map
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_providers_have_display_names() {
        let names = built_in_provider_display_names();
        assert_eq!(names.get("anthropic").map(String::as_str), Some("Anthropic"));
        assert_eq!(
            names.get("prime-inference").map(String::as_str),
            Some("Prime Inference")
        );
        assert_eq!(
            names.get("xiaomi-token-plan-sgp").map(String::as_str),
            Some("Xiaomi MiMo Token Plan (Singapore)")
        );
        // The bundled DGX gateway providers match the names a DGX Auth File
        // import writes into models.json.
        assert_eq!(names.get("dgx-glm53").map(String::as_str), Some("DGX GLM 5.3"));
        assert_eq!(names.get("dgx-k3").map(String::as_str), Some("DGX K3"));
        assert_eq!(
            names.get("dgx-deepseek41").map(String::as_str),
            Some("DGX DeepSeek 4.1")
        );
    }
}
