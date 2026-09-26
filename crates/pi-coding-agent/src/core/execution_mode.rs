//! The model-facing execution surface, independent of presentation and autonomy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutionMode {
    #[default]
    Ipython,
    Direct,
}

impl ExecutionMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim() {
            "ipython" => Ok(Self::Ipython),
            "direct" => Ok(Self::Direct),
            _ => Err("Usage: /mode [ipython|direct|toggle]".into()),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ipython => "ipython",
            Self::Direct => "direct",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Ipython => "IPython",
            Self::Direct => "Direct tools",
        }
    }

    pub fn toggled(self) -> Self {
        match self {
            Self::Ipython => Self::Direct,
            Self::Direct => Self::Ipython,
        }
    }

    pub fn from_tools(tools: &[String]) -> Option<Self> {
        if tools.iter().any(|name| name == "ipython") {
            Some(Self::Ipython)
        } else if ["bash", "edit"]
            .iter()
            .all(|name| tools.iter().any(|tool| tool == name))
        {
            Some(Self::Direct)
        } else {
            None
        }
    }

    pub fn tools(self, current: &[String]) -> Vec<String> {
        let mut tools: Vec<_> = current
            .iter()
            .filter(|name| !matches!(name.as_str(), "ipython" | "bash" | "edit"))
            .cloned()
            .collect();
        tools.extend(match self {
            Self::Ipython => vec!["ipython".into()],
            Self::Direct => vec!["bash".into(), "edit".into()],
        });
        tools
    }
}
