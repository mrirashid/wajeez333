function logintoHome(event) {
    // Prevent the form submission
    event.preventDefault(); 
    // Redirect to the Flask profile route
    window.location.href = '/profile'; 
}

const casesData = {
    "cases": [
        {
            "type": "جنائية",
            "summary": "قضية سرقة منزل في حي السلام",
            "date": "2024-01-15"
        },
        {
            "type": "جنائية",
            "summary": "حادث مروري في شارع الملك فهد",
            "date": "2024-02-20"
        },
        {
            "type": "أحوال شخصية",
            "summary": "دعوى نفقة وحضانة",
            "date": "2024-01-10"
        },
        {
            "type": "أحوال شخصية",
            "summary": "طلب تعديل زيارة أطفال",
            "date": "2024-03-05"
        },
        {
            "type": "عام",
            "summary": "نزاع تجاري بين شركتين",
            "date": "2024-02-01"
        }
    ]
};
function processData(data) {
    const counts = {
        "جنائية": 0,
        "أحوال شخصية": 0,
        "عام": 0
    };
    
    data.cases.forEach(caseItem => {
        counts[caseItem.type]++;
    });

    return {
        labels: Object.keys(counts),
        data: Object.values(counts)
    };
}

// Create doughnut chart
function createDoughnutChart(processedData) {
    const ctx = document.getElementById('caseChart').getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: processedData.labels,
            datasets: [{
                data: processedData.data,
                backgroundColor: ['#003366', '#36A2EB', '#8dc9e5'],
                hoverBackgroundColor: ['#40372E', '#A88A6C', '#ADA195'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: { size: 14 }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.formattedValue;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = Math.round((context.raw / total) * 100);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            },
            cutout: '60%'
        }
    });
}

// Display summaries
function displaySummaries(data) {
    const summariesContainer = document.getElementById('caseSummaries');
    data.cases.forEach(caseItem => {
        const summaryDiv = document.createElement('div');
        let className = '';
        switch(caseItem.type) {
            case 'جنائية':
                className = 'criminal';
                break;
            case 'أحوال شخصية':
                className = 'personal';
                break;
            case 'عام':
                className = 'general';
                break;
        }
        summaryDiv.className = `case-summary ${className}`;
        summaryDiv.innerHTML = `
            <strong>${caseItem.type}</strong>
            <p>${caseItem.summary}</p>
            <small>${new Date(caseItem.date).toLocaleDateString('ar-SA')}</small>
        `;
        summariesContainer.appendChild(summaryDiv);
    });
}

document.addEventListener('DOMContentLoaded', function() {
    const doughnutData = processData(casesData);
    const totalDocuments = casesData.cases.length; // Calculate total number of documents (cases)

    // Create doughnut chart
    createDoughnutChart(doughnutData);

    // Display the total number of documents
    document.getElementById('totalDocuments').innerHTML = `إجمالي عدد القضايا: ${totalDocuments}`;

    // Display case summaries
    displaySummaries(casesData);
});