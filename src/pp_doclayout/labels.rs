use serde::Serialize;
use std::str::FromStr;

pub const PP_DOCLAYOUT_V3_LABELS: [PPDocLayoutV3Label; 25] = [
    PPDocLayoutV3Label::Abstract,
    PPDocLayoutV3Label::Algorithm,
    PPDocLayoutV3Label::AsideText,
    PPDocLayoutV3Label::Chart,
    PPDocLayoutV3Label::Content,
    PPDocLayoutV3Label::DisplayFormula,
    PPDocLayoutV3Label::DocTitle,
    PPDocLayoutV3Label::FigureTitle,
    PPDocLayoutV3Label::Footer,
    PPDocLayoutV3Label::FooterImage,
    PPDocLayoutV3Label::Footnote,
    PPDocLayoutV3Label::FormulaNumber,
    PPDocLayoutV3Label::Header,
    PPDocLayoutV3Label::HeaderImage,
    PPDocLayoutV3Label::Image,
    PPDocLayoutV3Label::InlineFormula,
    PPDocLayoutV3Label::Number,
    PPDocLayoutV3Label::ParagraphTitle,
    PPDocLayoutV3Label::Reference,
    PPDocLayoutV3Label::ReferenceContent,
    PPDocLayoutV3Label::Seal,
    PPDocLayoutV3Label::Table,
    PPDocLayoutV3Label::Text,
    PPDocLayoutV3Label::VerticalText,
    PPDocLayoutV3Label::VisionFootnote,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PPDocLayoutV3Label {
    Abstract,
    Algorithm,
    AsideText,
    Chart,
    Content,
    DisplayFormula,
    DocTitle,
    FigureTitle,
    Footer,
    FooterImage,
    Footnote,
    FormulaNumber,
    Header,
    HeaderImage,
    Image,
    InlineFormula,
    Number,
    ParagraphTitle,
    Reference,
    ReferenceContent,
    Seal,
    Table,
    Text,
    VerticalText,
    VisionFootnote,
}

impl PPDocLayoutV3Label {
    /// Returns the number of class labels emitted by the PP-DocLayoutV3 checkpoint.
    pub const fn class_count() -> usize {
        PP_DOCLAYOUT_V3_LABELS.len()
    }

    /// Returns the official snake_case label name used for serialization.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Abstract => "abstract",
            Self::Algorithm => "algorithm",
            Self::AsideText => "aside_text",
            Self::Chart => "chart",
            Self::Content => "content",
            Self::DisplayFormula => "display_formula",
            Self::DocTitle => "doc_title",
            Self::FigureTitle => "figure_title",
            Self::Footer => "footer",
            Self::FooterImage => "footer_image",
            Self::Footnote => "footnote",
            Self::FormulaNumber => "formula_number",
            Self::Header => "header",
            Self::HeaderImage => "header_image",
            Self::Image => "image",
            Self::InlineFormula => "inline_formula",
            Self::Number => "number",
            Self::ParagraphTitle => "paragraph_title",
            Self::Reference => "reference",
            Self::ReferenceContent => "reference_content",
            Self::Seal => "seal",
            Self::Table => "table",
            Self::Text => "text",
            Self::VerticalText => "vertical_text",
            Self::VisionFootnote => "vision_footnote",
        }
    }

    /// Returns a deterministic RGBA color used for debug page annotations.
    pub fn debug_color_rgba(self) -> [u8; 4] {
        match self {
            Self::Abstract => [0x7C, 0x4D, 0xFF, 255],
            Self::Algorithm => [0x5C, 0x6B, 0xC0, 255],
            Self::AsideText => [0x26, 0xA6, 0x9A, 255],
            Self::Chart => [0xEF, 0x6C, 0x00, 255],
            Self::Content => [0x43, 0xA0, 0x47, 255],
            Self::DisplayFormula => [0xAB, 0x47, 0xBC, 255],
            Self::DocTitle => [0xD8, 0x1B, 0x60, 255],
            Self::FigureTitle => [0x00, 0x89, 0x7B, 255],
            Self::Footer => [0x8D, 0x6E, 0x63, 255],
            Self::FooterImage => [0xA1, 0x88, 0x7F, 255],
            Self::Footnote => [0xF4, 0x43, 0x36, 255],
            Self::FormulaNumber => [0x8E, 0x24, 0xAA, 255],
            Self::Header => [0xFF, 0x8F, 0x00, 255],
            Self::HeaderImage => [0xF9, 0xA8, 0x25, 255],
            Self::Image => [0x9E, 0x9E, 0x9E, 255],
            Self::InlineFormula => [0x7B, 0x1F, 0xA2, 255],
            Self::Number => [0x39, 0x49, 0xAB, 255],
            Self::ParagraphTitle => [0x1E, 0x88, 0xE5, 255],
            Self::Reference => [0x00, 0xAC, 0xC1, 255],
            Self::ReferenceContent => [0x00, 0x96, 0x88, 255],
            Self::Seal => [0xC6, 0x28, 0x28, 255],
            Self::Table => [0x00, 0x96, 0x88, 255],
            Self::Text => [0x43, 0xA0, 0x47, 255],
            Self::VerticalText => [0x6D, 0x4C, 0x41, 255],
            Self::VisionFootnote => [0xEC, 0x40, 0x7A, 255],
        }
    }
}

impl TryFrom<usize> for PPDocLayoutV3Label {
    type Error = PPDocLayoutV3LabelError;

    /// Converts a zero-based class id into its PP-DocLayoutV3 label.
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        PP_DOCLAYOUT_V3_LABELS
            .get(value)
            .copied()
            .ok_or(PPDocLayoutV3LabelError::UnknownClassId(value))
    }
}

impl FromStr for PPDocLayoutV3Label {
    type Err = PPDocLayoutV3LabelError;

    /// Parses an official snake_case label name.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        PP_DOCLAYOUT_V3_LABELS
            .iter()
            .copied()
            .find(|label| label.as_str() == value)
            .ok_or_else(|| PPDocLayoutV3LabelError::UnknownLabel(value.to_string()))
    }
}

impl std::fmt::Display for PPDocLayoutV3Label {
    /// Formats a label as its official snake_case name.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PPDocLayoutV3LabelError {
    #[error("unknown PP-DocLayoutV3 class id: {0}")]
    UnknownClassId(usize),
    #[error("unknown PP-DocLayoutV3 label: {0}")]
    UnknownLabel(String),
}
